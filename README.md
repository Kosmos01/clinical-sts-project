# clinical-sts-project

Project goal: Within the health informatics domain, we leverage semenatic textual similarity (STS) to identify similar clinical texts that can be found in Electronic Health Records. Similarity will be on a scale from 0 to 5, with 0 being completely dissimilar and 5 being completely similar.

Approach: Utilize medical entity identifiers (cTAKES, CLAMP), feature engineer, word embedding models, contextual representations, along with five regression/classification algorithms

Significance: It is imperative to enable machines to not only score how similar texts are, but to also have scores that correlate with clinicians' identifications so as to preserve any critical information

## Similarity examples (format: Sentence1 ---- Sentence2 ----> similarityScore):

> The patient reports no suicidal or homicidal ideation. ---- The patient reports agreement with the plan with no further questions at this time -> 0

> -- Colace 100 mg tablet, take one tablet twice daily as needed to keep stools soft. ---- losartan-HCTZ [HYZAAR] 100-12.5mg tablet 1 tablet by mouth one time daily ----> 1

> male who presents today for a skin examination. ---- female who presents for the follow up of sero negative rheumatoid arthritis ----> 2

> ibuprofen [ADVIL] 200 mg tablet 2 tablets by mouth as needed. ---- Tylenol Extra Strength 500 mg tablet 2 tablets by mouth every 6 hours as needed. ----> 3

> Musculoskeletal:  Normal passive and active range of motion. ---- TRight foot: There is normal range of motion and no deformity. ----> 4

> The patient has no further questions or concerns at this time. ---- She has no further questions or concerns at this time. ----> 5

*Note that in the actual dataset annotations were done by clinicians and would have intermediate values such as 1.22, 2.3, 3.1, 3.4 as similarity scores*

Scripts modified and implemented from others' github code: 
tokenization.py, modeling.py, extract_features.py, berts.py  
