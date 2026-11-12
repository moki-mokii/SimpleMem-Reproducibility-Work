# Evidence for two LoCoMo pairs

Both pairs come from round 3 of `conv-43` in the fixed [question manifest](questions.json). The dialogue turn IDs refer to the [pinned LoCoMo dataset](https://github.com/snap-research/locomo/blob/cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc/data/locomo10.json). The memory IDs identify entries in the run's frozen snapshot. The tables show the entries discussed in the report.

## Question 38

**Question** — which country has Tim visited most frequently in his travels?

**Reference answer** — UK

| Reflection | Initial entries | Added entries | Recorded answer | Token F1 | API tokens |
| --- | ---: | ---: | --- | ---: | ---: |
| Enabled | 40 | 16 | The UK | 0.666667 | 19,665 |
| Disabled | 28 | 0 | Cannot be determined from the provided context | 0.000000 | 3,779 |

| Where retrieved | Memory ID | Content of the memory |
| --- | --- | --- |
| Initial retrieval in both runs | `515de3d2-4d11-41ad-8580-f68ac8af7c8e` | Tim read about UK castles and wanted to visit them. |
| Enabled initial retrieval | `467c2914-7b46-403d-9577-7df398191efe` | Tim had visited a castle on a UK trip (D18:1). |
| Enabled supplemental retrieval | `c733f580-1a8e-4cff-81bf-ee4eedbe98d1` | Tim had attended a Harry Potter conference in the UK on an earlier trip (D13:1). |

The enabled answer had evidence of two trips, one from each retrieval stage. The initial lists differed, so the F1 difference combines changes in the initial and supplemental evidence.

## Question 106

**Question** — Where are John and his teammates planning to explore on a team trip?

**Reference answer** — a new city

| Reflection | Initial entries | Added entries | Recorded answer | Token F1 | API tokens |
| --- | ---: | ---: | --- | ---: | ---: |
| Enabled | 31 | 39 | They are still deciding; Edinburgh, Scotland has been suggested and is being considered. | 0.000000 | 19,856 |
| Disabled | 31 | 0 | A new city, but the destination is still being decided | 0.461538 | 3,831 |

| Where retrieved | Memory ID | Content of the memory |
| --- | --- | --- |
| Initial retrieval in both runs | `c7b841ca-54eb-4d69-a040-b86cab612213` | The team planned to explore a new city (D11:7). |
| Enabled supplemental retrieval | `2049c79e-59f0-4511-aedc-2574fdb8b675` | Tim suggested Edinburgh (D11:10). |
| Enabled supplemental retrieval | `ccd564e0-949d-4864-aba4-3501b562e54f` | John said Edinburgh was worth considering (D11:11). |

The two runs began with the same 31 entries in the same order. The enabled answer reported a suggestion grounded in the dialogue and shared zero tokens with the reference. The disabled answer included all three reference tokens.
