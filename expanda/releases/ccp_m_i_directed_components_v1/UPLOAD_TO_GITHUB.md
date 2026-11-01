# 上传到 GitHub

## 推荐方式：上传当前组合源码

解压 `ccp_m_i_directed_components_v1_source.tar.gz` 后，进入：

```text
ccp_m_i_directed_components_v1/current_source
```

该目录保持原始仓库路径结构，可作为一个新 GitHub 仓库的内容。建议先运行：

```bash
python3 expanda/releases/ccp_i_directed_balanced_v1/verify_release.py
```

然后初始化并上传：

```bash
git init
git add .
git commit -m "add frozen CCP directed M and I component sources"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## 保存 M 的精确历史版本

`frozen_m_tag` 是 M 标签 `ccp-m-directed-balanced-v1` 对应的精确文件快照。可以将它作为仓库中的 `archive/frozen_m_tag` 保存，也可以单独建立一个历史分支。

验证 M 快照：

```bash
cd frozen_m_tag
python3 expanda/releases/ccp_m_directed_balanced_v1/verify_release.py
```

## 不要上传为正式结果的配置

不要把以下配置描述为最终 I 方法：

```text
joint_balanced_hv_gate
```

它只作为失败实验记录保留。正式 I 配置是：

```text
joint_balanced
```

正式 M 配置是：

```text
REPLACE_G10_JOINT_BALANCED
```
