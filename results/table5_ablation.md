# Table 5: Ablation Study — Feature Contribution (NDCG@10)

| Dropped Feature | HR@10 | NDCG@10 | Δ NDCG@10 |
|----------------|-------|---------|-----------|
| None (full model) | 0.7175 | 0.4325 | — |
| ease | 0.7029 | 0.4113 | -0.0212 |
| neumf | 0.7114 | 0.4327 | +0.0002 |
| knn | 0.7135 | 0.4338 | +0.0013 |
| dcn | 0.7181 | 0.4338 | +0.0013 |
| lgcn | 0.7205 | 0.4347 | +0.0022 |
| bpr | 0.7188 | 0.4349 | +0.0024 |
| pop | 0.7211 | 0.4358 | +0.0033 |
| llm_yes_prob | 0.7196 | 0.4360 | +0.0035 |

*Negative Δ means the feature hurts when removed.*