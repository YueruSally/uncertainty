"""S11 rule trajectories and frozen residual selection; audit labels never steer search."""
from common import *
from copy import deepcopy
from types import SimpleNamespace
import random,time,resource
import numpy as np
import baseline_uncertainty as base
from multi_instance_logging import fingerprint
from transformer_ppo.comparison_controllers import BaseBudgetController
from transformer_ppo.controller import PPOController
from initialization import (InitializationK8Controller,individual_batch_tokens,
    candidate_proxy_features,deterministic_proxy_rows)
from crossover import (CrossoverK8Controller,_pair_proxy_rows,_pair_batch_tokens,
    _pair_candidate_features)
from mutation import build_replace_pool,batch_tokens
from m_candidate_information import proposal_randomness,private_seed,proposal_features
from transformer_ppo.archive import exact_hv_3d

def fixed_seed_count(population):
    seeds=int(round(population*base.FEASIBILITY_SEED_FRACTION))
    if base.FEASIBILITY_SEED_FRACTION>0 and seeds==0:seeds=1
    seeds=min(population,max(0,seeds));remaining=population-seeds
    return min(population,seeds+3*(max(1,remaining//6) if remaining else 0))


def guarded_reference(rows,legacy,base_index,tolerance):
    """Deterministic proxy only; this is not a true-feasibility or HV guarantee."""
    proposed,baseline=rows[legacy],rows[base_index]
    for key in ('missing','capacity_excess','lateness'):
        if proposed[key]>baseline[key]+1e-10:return base_index
    for key in ('cost','emission','time'):
        if proposed[key]>(1+tolerance)*baseline[key]+1e-10:return base_index
    return legacy


def mutation_proxy_features(rows):
    features=candidate_proxy_features(rows)
    # M order is Balanced, Base, replacement variants; do not reuse I role flags.
    for i,values in enumerate(features):values[-4:]=[float(i==1),float(i==0),float(i>=2),0.]
    return features


def timed_features(fn):
    def call(self,*args,**kwargs):
        start=time.perf_counter()
        try:return fn(self,*args,**kwargs)
        finally:self._feature_seconds+=time.perf_counter()-start
    return call

EVAL_FIELDS=tuple(k for k in base.Individual.__dataclass_fields__ if k not in ('od_allocations','rank','crowding_distance'))
class ScreenController(BaseBudgetController):
    def __init__(self,*args,settings,arm,collect=None,learned=None,reliable_options,batches,arcs,**kwargs):
        self.settings=settings;self.arm=arm;self.collect=collect;self.learned=learned
        self.controls_initialization=arm.startswith('I_') or collect=='I'
        self.controls_crossover=arm.startswith('C_') or collect=='C'
        self.controls_mutation=arm.startswith('M_') or collect=='M'
        self.rule_profile=settings.get('rule_profile','LEGACY')
        self.rule_counts={'proxy_fallback_to_base':0,'fixed_initial_slots_preserved':0}
        self.i_start_slot=fixed_seed_count(settings['population']) if self.rule_profile=='GUARDED' else 0
        self.initial_fingerprints=[]
        self.batches=tuple(batches);self.arcs=arcs;self.reliable_options=reliable_options
        self.population=[];self.pending_children=[];self.cache={};self.raw_points=[];self.feedback=[]
        self.audit_counts={k:0 for k in 'ICM'};self.audit_seconds=0.;self.audit_seconds_by_component={k:0. for k in 'ICM'};self.audit_states={k:0 for k in 'ICM'}
        self.selection={k:{'calls':0,'fallbacks':0,'switches':0,'unsupported':0} for k in 'ICM'}
        self.slot=0;self.c_event=0;self.m_event=0;self.pending_i=None;self.pending_c=None
        self.m_start=None;self.checkpoints=[];self._last_count=0;self.stalled=0;self.stop_reason=None
        self.phase_counts={k:[0,0,0] for k in 'ICM'}
        self._feature_seconds=0.;self.evaluation_trace=[]
        self.i_slots=set(np.linspace(self.i_start_slot,settings['population']-1,min(settings['states']['I'],settings['population']-self.i_start_slot),dtype=int).tolist())
        self.targets={'C':np.linspace(settings['population'],settings['budget']-50,settings['states']['C']).tolist()}
        super().__init__(*args,**kwargs)
        # Candidate construction only. These helpers never evaluate or commit a search decision.
        self.ih=InitializationK8Controller.__new__(InitializationK8Controller)
        self.ih.reliable_options=reliable_options;self.ih.run_id=self.run_id;self.ih.pool_size=8
        self.ih._active=None;self.ih.pool_events=0
        self.ch=CrossoverK8Controller.__new__(CrossoverK8Controller)
        self.ch.run_id=self.run_id;self.ch.pool_size=8;self.ch._active=None
        self.ch.total_attempts=self.ch.pool_events=self.ch.no_candidate=self.ch.eligible_pool_events=self.ch.unique_candidate_total=0
        self.mh=PPOController.__new__(PPOController);self.mh.mutation_control='operator'
        self.mh.run_id=self.run_id;self.mh.seeds=self.seeds

    def has_budget(self,required):
        if self.collect=='I' and self.generation>=0:
            self.stop_reason='initialization_complete';return False
        return self.evaluations+required<=self.budget

    def _due(self,component):
        n=self.audit_states[component]
        if component=='I':return self.slot in self.i_slots
        return n<len(self.targets.get(component,[])) and self.evaluations>=self.targets[component][n]

    @property
    def feature_construction_seconds(self):return self._feature_seconds

    def _state(self,component,pairs,features,tokens,reference):
        order=[reference]+[i for i in range(len(pairs)) if i!=reference]
        pairs=[pairs[i] for i in order];identities=[tuple(fingerprint(x) for x in pair) for pair in pairs]
        seen=set();mask=[]
        for key in identities:mask.append(key not in seen);seen.add(key)
        return pairs,{'feature_version':FEATURE_VERSION,'component':component,'instance':self.instance['instance_id'],
            'configuration':self.instance['configuration'],'evaluation':self.evaluations,'generation':self.generation,
            'progress':self.evaluations/self.settings['progress_denominator'],
            'initialization_progress':self.slot/self.settings['population'] if component=='I' else 0.,
            'population_fraction':len(self.population)/self.settings['population'],
            'population_raw':[safe_objectives(x) for x in self.population if x.feasible],
            'divisor':divisor(self.instance),'candidate_features':[features[i] for i in order],
            'candidate_batch_tokens':[tokens[i] for i in order],'legal_mask':mask,
            'fingerprints':[list(k) for k in identities],'reference_original_index':reference}

    def _select(self,component,row):
        stats=self.selection[component];stats['calls']+=1
        chosen=0
        if self.arm.endswith('LEARN'):
            start=time.perf_counter();chosen=self.learned.select(row);self.inference_seconds+=time.perf_counter()-start
            stats['fallbacks']+=int(chosen==0)
        stats['switches']+=int(chosen!=0)
        return chosen

    def _audit(self,component,pairs,row,function,args,kwargs):
        cap=self.settings['extra_label_budget_per_component_instance']
        if self.audit_counts[component]+sum(len(p) for p in pairs)>cap:return
        before=self.evaluations;py=random.getstate();npstate=np.random.get_state();start=time.perf_counter()
        memo={};labels=[];new=0
        with proposal_randomness(private_seed('audit',self.run_id,component,self.audit_states[component])):
            for pair in pairs:
                group=[]
                for original in pair:
                    key=fingerprint(original)
                    if key not in memo:
                        child=deepcopy(original)
                        if key in self.cache:
                            for k,v in self.cache[key].items():setattr(child,k,deepcopy(v))
                        else:function(child,*args,**kwargs);new+=1
                        memo[key]={'objectives':safe_objectives(child),'feasible':bool(child.feasible),'fingerprint':key}
                    group.append(memo[key])
                labels.append(group)
        assert random.getstate()==py and self.evaluations==before
        assert all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b for a,b in zip(npstate,np.random.get_state()))
        self.audit_counts[component]+=new;elapsed=time.perf_counter()-start;self.audit_seconds+=elapsed;self.audit_seconds_by_component[component]+=elapsed
        if component=='I':fraction=(self.slot-self.i_start_slot)/max(1,self.settings['population']-1-self.i_start_slot)
        else:
            start_at=self.m_start if component=='M' else self.settings['population']
            fraction=(self.evaluations-start_at)/max(1,self.budget-start_at)
        phase=max(0,min(2,int(fraction*3)))
        self.phase_counts[component][phase]+=1;self.audit_states[component]+=1
        self.feedback.append({**row,'labels':labels,'additional_evaluations':new,'phase':phase})
        print(f'[S11-TARGET-LABEL] {component} states={self.audit_states[component]} extra={self.audit_counts[component]}/{cap}',flush=True)

    def evaluate(self,function,ind,*args,**kwargs):
        if self.pending_i and id(ind)==self.pending_i[0]:
            _,pairs,row=self.pending_i;self.pending_i=None
            if self.collect=='I' and self._due('I'):self._audit('I',pairs,row,function,args,kwargs)
            chosen=self._select('I',row);new_fields=deepcopy(pairs[chosen][0].__dict__);ind.__dict__.clear();ind.__dict__.update(new_fields)
        key=fingerprint(ind)
        if getattr(ind,'_learning_eval_key',None)==(self.scenario_id,key):
            for k,v in self.cache[key].items():setattr(ind,k,deepcopy(v))
            self.cache_hits+=1
        else:
            if self.evaluations>=self.budget:raise RuntimeError('Search budget exhausted')
            started=time.perf_counter();function(ind,*args,**kwargs);self.objective_evaluation_seconds+=time.perf_counter()-started
            self.evaluations+=1
            self.evaluation_trace.append(key)
            self.cache[key]={k:deepcopy(getattr(ind,k)) for k in EVAL_FIELDS}
            if ind.feasible:self.raw_points.append(safe_objectives(ind))
        ind._learning_eval_key=(self.scenario_id,key);ind._learning_eval_id=self.evaluations
        if self.generation<0:
            if not any(x is ind for x in self.population):self.population.append(ind)
        elif not any(x is ind for x in self.pending_children):self.pending_children.append(ind)
        if self.evaluations in self.settings['milestones'] and not any(r['evaluation']==self.evaluations for r in self.checkpoints):
            candidates=deepcopy(self.population+self.pending_children)
            selected=base.nsga2_environmental_selection(candidates,min(len(candidates),self.settings['population']))
            self.checkpoints.append({'evaluation':self.evaluations,'generation':self.generation,
                'hv':self._hv(selected),'definition':'read-only environmental selection over current population plus evaluated pending offspring'})

    def _hv(self,population):
        return exact_hv_3d([self.normalizer.transform(x.objectives) for x in population if x.feasible],self.normalizer.reference)

    @timed_features
    def initial_individual_from_base(self,ind,batches,path_lib,max_paths=3,slot_index=None):
        self.slot=int(slot_index)
        if self.slot<self.i_start_slot:
            self.rule_counts['fixed_initial_slots_preserved']+=1
            return ind
        self.ih.pool_events=self.slot;self.ih._active=None
        if self.collect=='I' and not self._due('I'):
            # Construct only the Directed candidate on unsampled prefix slots.
            chosen=deepcopy(ind)
            for batch in batches:
                key=(batch.origin,batch.destination,batch.batch_id);old=ind.od_allocations.get(key,[])
                paths=list(path_lib.get((batch.origin,batch.destination),()))
                if not paths:continue
                k=min(len(old),max_paths,len(paths)) if old else 1
                ranked=self.ih._ranked_paths(paths,batch,'joint_balanced',self.slot)[:k]
                shares=[x.share for x in old[:k]] if old else [1.]
                chosen.od_allocations[key]=base.merge_and_normalize([base.PathAllocation(p,float(s)) for p,s in zip(ranked,shares)])
            return chosen
        placeholder=self.ih.initial_individual_from_base(ind,batches,path_lib,max_paths,slot_index)
        candidates=self.ih._active['candidates'];self.ih._active=None
        proxy=deterministic_proxy_rows(candidates,batches,self.arcs)
        pairs,row=self._state('I',[[x] for x in candidates],candidate_proxy_features(proxy),[individual_batch_tokens(x,batches) for x in candidates],1)
        self.pending_i=(id(placeholder),pairs,row);return placeholder

    @timed_features
    def crossover_common_node(self,p1,p2,batches,tt_dict,arc_lookup):
        self.c_event+=1
        self.ch._active=None;self.ch.generation=self.generation
        pair=self.ch.crossover_common_node(p1,p2,batches,tt_dict,arc_lookup);self.pending_c=self.ch._active
        return pair

    def process_crossover_pair(self,function,c1,c2,batches,path_lib,tt_dict,arc_lookup,arcs,wcost,wemis,reliable_options=None,paired_mutation_rate=None,**kwargs):
        active=self.pending_c;self.pending_c=None
        if not active:return None
        if (c1 is not active['pairs'][0][0] or c2 is not active['pairs'][0][1]):return None
        pairs=active['pairs'];build_started=time.perf_counter()
        def construct():
            for pos in (0,1):self.ch._paired_base_mutation(pairs,pos,batches,path_lib,tt_dict,arc_lookup,reliable_options,paired_mutation_rate)
            for pair in pairs:
                for child in pair:base.repair_missing_allocations(child,batches,path_lib)
        construct()
        proxy=_pair_proxy_rows(pairs,batches,arcs)
        ref=min(range(8),key=lambda i:(proxy[i]['score'],i!=1,i))
        if self.rule_profile=='GUARDED':
            protected=guarded_reference(proxy,ref,0,self.settings['guard_max_relative_worsening'])
            self.rule_counts['proxy_fallback_to_base']+=int(protected!=ref);ref=protected
        pairs,row=self._state('C',pairs,_pair_candidate_features(proxy,active['roles']),[_pair_batch_tokens(p,batches) for p in pairs],ref)
        self._feature_seconds+=time.perf_counter()-build_started
        args=(batches,arcs,tt_dict,wcost,wemis)
        if self.collect=='C' and self._due('C'):
            self._audit('C',pairs,row,function,args,kwargs)
        chosen=self._select('C',row)
        for child in pairs[chosen]:self.evaluate(function,child,*args,**kwargs)
        return pairs[chosen]

    @timed_features
    def _m_pool(self,ind,base_child,batches,path_lib,tt_dict,arc_lookup,reliable_options):
        self.mh.generation=self.generation;self.mh._active_mixture_event_index=self.m_event
        proposals=build_replace_pool(self.mh,ind,batches,path_lib,tt_dict,arc_lookup,reliable_options)
        eligible=bool(proposals[0]['eligible'])
        # Keep Balanced and the actually pre-drawn Base mutation in the same K8 pool.
        balanced=proposals[0]['child'] if eligible else deepcopy(base_child)
        candidates=[balanced,base_child]+[p['child'] for p in proposals[1:7]]
        while len(candidates)<8:candidates.append(deepcopy(base_child))
        proxy=deterministic_proxy_rows(candidates,batches,self.arcs)
        ref=0 if eligible else 1
        if self.rule_profile=='GUARDED':
            protected=guarded_reference(proxy,ref,1,self.settings['guard_max_relative_worsening'])
            self.rule_counts['proxy_fallback_to_base']+=int(protected!=ref);ref=protected
        if not eligible:self.selection['M']['unsupported']+=1
        return self._state('M',[[x] for x in candidates],mutation_proxy_features(proxy),
            [individual_batch_tokens(x,batches) for x in candidates],ref)

    def mutate(self,ind,batches,path_lib,tt_dict,arc_lookup,arcs,wcost,wemis,reliable_options=None,**kwargs):
        self.m_event+=1;args=(batches,arcs,tt_dict,wcost,wemis)
        eligible=self.generation>=self.settings['m_start_generation']
        if eligible and self.m_start is None:
            self.m_start=self.evaluations
            self.targets['M']=np.linspace(self.m_start,max(self.m_start,self.budget-50),self.settings['states']['M']).tolist()
        if eligible and self.controls_mutation:
            # Draw the ordinary Base mutation exactly once. Never evaluate it to select a candidate.
            base_child=deepcopy(ind);batch=random.choice(batches);op=base.sample_operator()
            ok=base.apply_mutation_op(base_child,op,batch,path_lib,tt_dict,arc_lookup,reliable_options=reliable_options)
            if ok:base.repair_missing_allocations(base_child,batches,path_lib)
            pairs,row=self._m_pool(ind,base_child,batches,path_lib,tt_dict,arc_lookup,reliable_options)
            if self.collect=='M' and self._due('M'):
                self._audit('M',pairs,row,base._evaluate_individual_impl,args,kwargs)
            selected=self._select('M',row)
            new_fields=deepcopy(pairs[selected][0].__dict__);ind.__dict__.clear();ind.__dict__.update(new_fields)
            self.evaluate(base._evaluate_individual_impl,ind,*args,**kwargs)
            return 'replace',True
        # Base before activation and in the other single-component experiments.
        batch=random.choice(batches);op=base.sample_operator()
        ok=base.apply_mutation_op(ind,op,batch,path_lib,tt_dict,arc_lookup,reliable_options=reliable_options)
        if ok:
            base.repair_missing_allocations(ind,batches,path_lib);self.evaluate(base._evaluate_individual_impl,ind,*args,**kwargs)
        return op,bool(ok)

    def flush_events(self,population):self.population=list(population);self.pending_children=[]
    def log_initial_population(self,population):
        self.initial_fingerprints=[fingerprint(x) for x in population]
        self.initial_evaluation_digest=canonical_digest(self.evaluation_trace);self.flush_events(population)
    def log_generation(self,population,boost):
        self.generations_completed+=1
        self.stalled=self.stalled+1 if self.evaluations==self._last_count else 0;self._last_count=self.evaluations
        if self.stalled>=self.settings['max_stalled_generations']:raise RuntimeError('No new evaluations for repeated generations; incomplete run preserved')
        write_json(self.out/'PROGRESS.json',{'search_evaluations':self.evaluations,'audit_evaluations':self.audit_counts,'generation':self.generation,'elapsed_seconds':time.perf_counter()-self.start})
        self.append('generation_summary',{'generation':self.generation,'evaluations':self.evaluations,'feasible':sum(x.feasible for x in population),'elapsed_seconds':time.perf_counter()-self.start})
    def finish_episode(self):pass
    def summary(self):
        return {**super().summary(),'rule_profile':self.rule_profile,'rule_counts':self.rule_counts,'initial_fingerprints':self.initial_fingerprints,'selection':self.selection,'audit_evaluations':self.audit_counts,
            'audit_seconds_by_component':self.audit_seconds_by_component,'audit_states':self.audit_states,'audit_phase_counts':self.phase_counts,'audit_seconds':self.audit_seconds,
            'cache_hits':self.cache_hits,'peak_rss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'milestone_hv':self.checkpoints,'final_feasible_fraction':sum(x.feasible for x in self.population)/max(1,len(self.population)),
            'stop_reason':'initialization_complete' if self.collect=='I' else ('evaluation_cap' if self.evaluations==self.budget else 'atomic_pair_budget_remainder'),
            'feature_construction_seconds':self._feature_seconds,
            'input_range_checks':self.learned.range_checks if self.learned else {},
            'evaluation_order_digest':canonical_digest(self.evaluation_trace),'initial_evaluation_digest':getattr(self,'initial_evaluation_digest',None),
            'hidden_candidate_objective_evaluations':0 if not self.collect else sum(self.audit_counts.values())}
