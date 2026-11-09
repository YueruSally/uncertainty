"""Rule-only counterpart of the historical deferred-mutation ICM controller."""
from combined import CombinedFrozenK8Controller, create_component_directories
from initialization import InitializationK8Controller
from crossover import CrossoverK8Controller, simple_index
from mutation import ScaleSearchController


class SimpleSelector:
    def select(self, row):
        # Exact historical Simple ordering and tie break; no learned model.
        return simple_index(row)


class DeferredSimpleC(CrossoverK8Controller):
    def __init__(self, *args, **kwargs):
        # The existing deferred API accepts a selector through transformer mode.
        # Supply the deterministic Simple selector, retaining the same pair flow.
        super().__init__(*args, selection_mode='transformer',
                         selection_policy=SimpleSelector(),
                         defer_selected_pair_mutation=True, **kwargs)

    def summary(self):
        return {**super().summary(), 'crossover_k8_selection_mode':'simple',
                'deferred_selector':'deterministic Simple; no network',
                'learned_selector':False}


class CombinedRuleController(CombinedFrozenK8Controller):
    def __init__(self, *args, reliable_options, arcs, **kwargs):
        batches=tuple(kwargs['batches']);normalizer=kwargs['archive'].normalizer
        ScaleSearchController.__init__(self,*args,**kwargs)
        root=create_component_directories(self.out)
        common=dict(method=self.method,instance=self.instance,seeds=self.seeds,normalizer=normalizer,
                    reliable_options=reliable_options,batches=batches,arcs=arcs,pool_size=8)
        self.i_controller=InitializationK8Controller(root/'I',self.run_id,self.scenario_id,self.budget,
                                                     selection_mode='balanced',**common)
        self.c_controller=DeferredSimpleC(root/'C',self.run_id,self.scenario_id,self.budget,**common)
        self._sync_children()

    def summary(self):
        return {**super().summary(),'combined_selector':'Directed + Simple + Balanced',
                'learned_components':[]}
