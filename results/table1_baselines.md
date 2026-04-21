# Table 1: Recommendation Performance on ML-1M (NCF leave-one-out protocol)

| Model | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|-------|-------|-------|-------|-------|-------|-------|
| Popularity | — | 0.4981 | — | — | 0.2768 | — |
| ItemKNN | — | 0.6810 | — | — | 0.3992 | — |
| BPR-MF | — | 0.6189 | — | — | 0.3477 | — |
| EASE^R | — | 0.6953 | — | — | 0.4296 | — |
| LightGCN | — | 0.4976 | — | — | 0.2757 | — |
| DCN-v2 | — | 0.6378 | — | — | 0.3591 | — |
| NeuMF | — | 0.6838 | — | — | 0.3942 | — |
| Hybrid (CF only) | 0.5390 | 0.7175 | 0.8681 | 0.3745 | 0.4325 | 0.4709 |
| --- |---|---|---|---|---|---|
| Hybrid (LambdaRank + Optuna) | **0.6360** | **0.7934** | **0.9087** | **0.5890** | **0.6403** | **0.6698** |

*Bold = best in column. — = not reported.*