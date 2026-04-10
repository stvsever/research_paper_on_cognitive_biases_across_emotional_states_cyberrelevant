## Record identity

**Title**
Cognitive Bias Vulnerability Across Emotional States in Cyber-Relevant Decision Contexts

**Registration type**
Preregistered protocol for a rapid review, taxonomy-construction study, and simulation-based evaluation study

**Status**
Public registration on OSF

**Registry**
OSF Registries. Created either from scratch ([OSF Support][1])

---

## Intended use

This registration specifies the rationale, scope, review procedures, taxonomy-construction rules, emotion-space specification, simulation design, model-judging framework, robustness analyses, and inferential criteria for a study examining how transient emotional states may shift susceptibility to specific cognitive biases in cyber-relevant decision contexts. The preregistration is intended to distinguish confirmatory from exploratory analyses, reduce analytic flexibility, and preserve a transparent decision trail across the literature review, taxonomy-building, and simulation phases.

---

## Citation

**Suggested citation**
Van Severen, S. (2026). *Cognitive Bias Vulnerability Across Emotional States in Cyber-Relevant Decision Contexts* [Preregistration]. OSF Registries.

---

# Review Methods

## Type of review

Rapid review with structured reporting, recursive backward and forward citation tracing, formal concept consolidation, and hierarchical taxonomy construction. The review is designed to identify, normalize, and organize named cognitive biases into a non-redundant hierarchical taxonomy specifically optimized for cyber-relevant decision contexts rather than for cognitive-bias theory in the abstract.

## Guidelines and tools used

PRISMA-style identification, screening, eligibility, and inclusion flow will be used for transparent reporting of the review pipeline. The review is rapid rather than exhaustive and is explicitly theory-building, taxonomy-oriented, and simulation-oriented rather than effect-size oriented.

## Review stages

1. Protocol finalization
2. Search development and validation
3. Database search execution
4. Record management and deduplication
5. Title and abstract screening
6. Full-text eligibility assessment
7. Concept extraction and normalization
8. Synonym consolidation and adjudication
9. Cyber-relevance tagging
10. Hierarchical taxonomy construction
11. Emotion lexicon finalization
12. Emotion-dimension calibration
13. Scenario, persona, and prompt calibration
14. LLM ensemble scoring
15. Robustness, sensitivity, and network analyses
16. Freeze of confirmatory outputs
17. Manuscript preparation

## Current review stage

Protocol development; formal searches not yet executed at the time of registration.

## Start date

2026-04-08

## End date

Planned completion: 2027-02-28

## Background

Cybersecurity governance and AI governance have advanced rapidly. NIST CSF 2.0 positions cybersecurity as an organization-wide risk management problem, NIST’s Generative AI Profile extends AI RMF guidance to generative systems, and OWASP’s 2025 LLM Top 10 frames AI-application security around a maturing risk landscape for deployed generative systems. Together, these frameworks show rapid maturation in technical and governance-layer thinking about cyber and AI risk. ([NIST][2])

By contrast, the human side of cyber risk remains less systematically organized at the level of transient cognition under pressure. Reviews of social engineering and cyberattack cognition describe the literature as superficial, scattered, and insufficiently systematized at the psychological level; recent systematic review work also finds that empirical social-engineering studies often cover only a small portion of the broader attack space, often lack a common reference structure for attack targetization, and frequently under-model context and cognitive processes. It is therefore reasonable to infer that there is still no broadly adopted, comprehensive hierarchical taxonomy of cognitive biases in cyber-relevant decision contexts, and even less work systematically linking such a taxonomy to higher-order emotional structure. ([Frontiers][3])

The present study addresses that gap by integrating four layers of representation. First, it constructs a non-redundant hierarchical taxonomy of cognitive biases relevant to cyber-relevant decision contexts. Second, it fixes an author-curated lexicon of 200+ emotional states. Third, it represents those emotions in multiple complementary forms: discrete labels, semantic clusters, continuous latent dimensions, and naturalistic multi-component states. Fourth, it estimates how these emotional representations relate to leaf-level bias susceptibility in standardized cyber-relevant decision situations. The goal is not operational manipulation, but a structured scientific account of where transient affect may increase, decrease, or reorganize modeled cognitive vulnerability.

A further motivation is methodological and ethical. Simulation-first analysis enables large-scale structured study without collecting sensitive real-world emotional or behavioral data in the core design. That aligns with the GDPR’s data-protection logic and fits more comfortably within the EU’s current risk-based AI regulatory environment than any live manipulation design would. ([EUR-Lex][4])

## Primary research question(s)

1. How can cognitive biases relevant to cyber-relevant decision contexts be organized into a comprehensive, non-redundant hierarchical taxonomy?
2. Which emotional states within the author-curated 200+ emotion lexicon are associated with increased or decreased susceptibility to each leaf-level cognitive bias, relative to a neutral affect reference state?
3. Which higher-order emotion clusters exhibit shared cognitive-bias vulnerability signatures?
4. Which cognitive-bias families exhibit shared affective vulnerability profiles?
5. Which latent emotional dimensions best explain variation in bias susceptibility across the hierarchy?
6. Which naturalistic multi-component emotional states produce non-additive shifts in bias susceptibility?

## Secondary research question(s)

1. Which individual emotions, emotion clusters, and emotion-dimension profiles appear relatively protective rather than vulnerability-enhancing?
2. Which regions of the cognitive-bias hierarchy are most affect-sensitive across scenario classes?
3. How stable are estimated associations across judge model families, prompt variants, synthetic persona profiles, and cyber-relevant scenario types?
4. Do projected emotion–bias networks reveal higher-order communities or vulnerability motifs not visible at the single-bias level?
5. To what extent can latent-dimensional models generalize to held-out or newly added emotion labels better than discrete-label-only models?

## Expectations / hypotheses

**H1.** Threat-, alarm-, and uncertainty-related emotional states will be associated with stronger susceptibility to biases involving ambiguity management, salience weighting, loss sensitivity, urgency-driven judgment, and authority-oriented heuristic reliance.

**H2.** Reward-, enthusiasm-, and approach-related emotional states will be associated with stronger susceptibility to biases involving optimism, reduced scrutiny, impulsive commitment, and asymmetric weighting of positive cues.

**H3.** Shame-, guilt-, dependency-, and social-self-evaluative emotional states will be associated with stronger susceptibility to conformity-, compliance-, reassurance-, and self-protective interpretive biases.

**H4.** Calm, reflective, skeptical, and safety-related emotional states will tend to attenuate at least a subset of fast, cue-driven, pressure-sensitive biases.

**H5.** Higher latent uncertainty and lower latent control will predict stronger susceptibility in bias families centered on ambiguity, urgency, and externally cued decision shortcuts.

**H6.** Higher other-oriented social appraisal profiles will predict stronger susceptibility in conformity-, compliance-, and relational-evaluation bias families.

**H7.** Naturalistic mixed states combining urgency, uncertainty, depletion, or social-evaluative pressure will yield stronger shifts in bias susceptibility than predicted by simple averaging of constituent single-emotion states.

---

## Dependent variable(s) / outcome(s) / main variables

### Primary outcome

**Emotion-Conditioned Bias Shift Score (ECBSS)**

**Definition**
For each emotion × bias leaf × scenario × persona × judge-model cell, ECBSS is the estimated signed shift in expected bias susceptibility relative to a neutral-affect reference condition.

**Scale**
−1000 to +1000

**Interpretation**

* **−1000** = maximal modeled attenuation of the focal bias
* **0** = no material modeled shift
* **+1000** = maximal modeled amplification of the focal bias

The final signed score will be computed in two stages:

1. **directional classification**: attenuation / no material shift / amplification
2. **magnitude rating**: 0–1000

The score will be treated as a high-resolution ordinal-analog scale, not as a literal interval psychometric unit. Primary inference will be based on directional consistency, anchored magnitude bands, and robustness across perturbations rather than on the isolated interpretation of any single raw integer.

### Secondary outcomes

* Directional shift label
* Magnitude band
* Judge confidence score
* Cross-model agreement index
* Prompt-stability index
* Persona-stability index
* Scenario-stability index
* Cluster-level mean shift
* Family-level mean shift
* Dimension-level effect estimates
* Network centrality metrics
* Composite-emotion interaction deviation score
* Reliability tier for each estimated cell

---

## Independent variable(s) / intervention(s) / treatment(s)

* Emotion condition: single-emotion state from the author-curated 200+ lexicon
* Emotion condition: naturalistic composite state
* Bias target: leaf-level cognitive bias
* Bias family: higher-order taxonomy branch
* Scenario type: cyber-relevant decision context
* Persona profile: synthetic persona specification
* Judge model family
* Prompt template variant
* Prompt order condition

## Additional variable(s) / covariate(s)

* Hierarchical depth of the bias node
* Taxonomy confidence tier
* Cyber-relevance evidence tier
* Emotion-cluster membership
* Emotion-vector location in embedding space
* Latent emotion coordinates: **V, A, C, U, S**
* Scenario ambiguity level
* Scenario time-pressure level
* Social-pressure level
* Authority cue density
* Perceived consequence severity
* Persona digital literacy
* Persona workload state
* Persona security familiarity
* Persona uncertainty tolerance
* Judge-family calibration class

---

## Emotion universe

The study will use the full author-curated 200+ emotion lexicon, fixed before scoring begins and archived as Appendix A in the OSF project. No emotion labels will be deleted post hoc. If near-synonyms are intentionally preserved by design, they will remain separate members of the scoring universe.

---

## Emotion representation strategy

The study will not treat emotion words as atomic units only. Each emotion will be represented in four parallel views:

1. **Discrete-label view**: the original emotion term
2. **Semantic-cluster view**: embedding-based clusters of related emotions
3. **Latent-dimension view**: continuous scores on V, A, C, U, and S
4. **Composite-state view**: naturalistic dyads and triads

This multi-view design is motivated by emotion research showing that valence and arousal remain important but are not sufficient on their own, that appraisal dimensions such as certainty and control differentiate emotions in systematic ways, that broader semantic spaces are richer than simple two-dimensional models, and that emotional categories are bridged by gradients rather than perfectly discrete boundaries. ([Of (im)possible interest][5])

### Latent emotional dimensions

Each emotion term will be mapped onto the following continuous latent dimensions:

* **V = valence**: negative ↔ positive
* **A = arousal**: low activation ↔ high activation
* **C = control / coping potential**: low perceived control ↔ high perceived control
* **U = uncertainty / expectedness**: expected/certain ↔ uncertain/unpredictable
* **S = social orientation / responsibility focus**: self-oriented ↔ other-oriented

This five-dimensional landing space is a preregistered operational synthesis, not a claim that one universally accepted factor analysis has established exactly these five dimensions as the single true structure of emotion. Rather, it integrates convergent traditions in affective science: valence/arousal from dimensional affect models, control/dominance and coping potential from appraisal and lexical norms work, certainty/expectedness from appraisal theory, and self–other responsibility or causal orientation from appraisal-social accounts. ([Of (im)possible interest][5])

### Dimension-score generation

Dimension scores will be generated using a multi-model LLM judging procedure with anchored rating instructions. For each emotion term, each judge model will rate V, A, C, U, and S using explicit anchor descriptions and contrastive exemplars. Where overlapping items exist in published valence–arousal–dominance lexical norms, those normative values will be used as calibration anchors for V, A, and C-related scaling. ([Springer][6])

For **V**, **A**, and **C**, calibration will rely on overlap with published word norms and with anchor emotion terms selected to span the scale. For **U** and **S**, calibration will rely on appraisal-theory anchor definitions, contrast sets, and within-ensemble convergence. Any item showing unstable dimension placement across judge families or prompt variants will be flagged as uncertain and carried forward with an uncertainty label rather than silently forced into a stable coordinate.

### Why the latent-dimension layer is included

The latent-dimension layer is included for three reasons. First, it reduces dependence on idiosyncratic wording. Second, it allows continuous modeling of vulnerability rather than only discrete emotion-by-emotion lookup. Third, it enables generalization tests: if dimensional models can predict held-out emotion profiles, then the results are less likely to be mere artifacts of individual word labels. This directly addresses a likely reviewer critique that emotion terms are too lexicalized or too atomistic to support general inference. ([jstor.org][7])

---

## Software

* Reference management: Zotero or equivalent
* Screening/extraction management: Rayyan, spreadsheet, or equivalent
* Analysis: Python and/or R
* Embedding, clustering, and dimensionality-reduction tools
* Network-analysis libraries
* Figure generation for UMAP, heatmaps, trees, and projected networks
* API-based multi-model orchestration tools
* OSF for versioning and archival

## Funding

No funding declared at registration.

## Conflicts of interest

No conflicts of interest declared at registration.

## Overlapping authorships

Single-author registration at submission stage: **Stijn Van Severen**

---

# Search Strategy

## Databases

* Scopus
* Web of Science Core Collection
* PsycINFO
* PubMed/MEDLINE
* ACM Digital Library
* IEEE Xplore
* Google Scholar for supplementary known-item retrieval and citation tracing

## Interfaces

Native platform interfaces where available. Google Scholar will be used for supplementary retrieval rather than as the primary record source.

## Grey literature

Grey literature will be used only when it contributes taxonomy-relevant conceptual material such as major handbooks, institutional reports, or canonical bias inventories that are frequently cited and not clearly superseded. Commercial blogs, vendor pages, general media commentary, and non-scholarly summaries will be excluded from the formal taxonomy corpus.

## Inclusion and exclusion criteria

### Inclusion criteria

* Review papers, systematic reviews, theoretical papers, handbooks, dictionaries, and seminal empirical papers that define, distinguish, catalogue, classify, compare, or elaborate named cognitive biases, heuristics, or systematic judgment distortions
* Sources relevant to human judgment, decision-making, reasoning, social cognition, trust, risk, uncertainty, compliance, misinformation, attention, interpretation, or memory
* Sources providing enough conceptual detail to support canonicalization, differentiation, hierarchical placement, or cyber-relevance tagging

### Exclusion criteria

* Sources using “bias” only in statistical, algorithmic, dataset, sampling, or fairness senses without substantive human cognitive content
* Purely machine-learning bias papers unrelated to human judgment
* Clinical cognitive distortions that are not clearly connected to the target cognitive-bias literature
* Commentaries, editorials, or abstracts without sufficient definitional content
* Sources focused only on intervention or mitigation while failing to define the underlying bias constructs
* Non-English sources unless clearly indispensable to a canonical concept

## Query strings

### Core conceptual block

`("cognitive bias*" OR "judgment bias*" OR "reasoning bias*" OR heuristic* OR "systematic error*" OR "decision-making bias*") AND (taxonomy OR taxonom* OR classif* OR hierarchy OR nomenclature OR review OR overview OR inventory OR catalog*)`

### Concept expansion block

`("cognitive bias*" OR heuristic* OR "judgment bias*") AND (definition OR mechanism OR typology OR distinction OR subtype OR synonym)`

### Cyber-relevance block

`("cognitive bias*" OR heuristic* OR "judgment bias*") AND (cybersecurity OR phishing OR "social engineering" OR misinformation OR trust OR compliance OR urgency OR authority OR uncertainty OR verification OR disclosure OR credibility)`

### Targeted recovery block

`("specific bias name") AND (definition OR taxonomy OR distinction OR mechanism OR example)`

## Search validation procedure

A benchmark set of known relevant review papers and high-yield conceptual sources will be specified before the final search is executed. Search strings will be iteratively refined until the benchmark set is recovered in at least one major database. Query adequacy will also be monitored with a concept-discovery curve to assess saturation of newly retrieved bias concepts.

## Other search strategies

* Backward citation tracing from included reviews and conceptual papers
* Forward citation tracing from highly central included sources
* Hand-searching of reference sections from high-yield sources
* Targeted searches for disputed synonyms, historical aliases, or unresolved concept splits

## Procedures to contact authors

Routine author contact is not planned. Authors may be contacted only if a concept-critical source cannot be obtained or if a taxonomically decisive definitional ambiguity cannot be resolved from accessible documents.

## Results of contacting authors

Not applicable at the time of registration.

## Search expiration and repetition

The search will be frozen after initial execution and a final update search will be run immediately prior to manuscript submission to identify recently published review or conceptual papers relevant to taxonomy completeness.

## Search strategy justification

The literature of interest spans psychology, cognitive science, behavioral decision-making, management, human factors, HCI, cybersecurity, social engineering, and misinformation research. A broad concept-sensitive retrieval strategy is therefore necessary. The objective is not pooled effect estimation but construction of a defensible conceptual structure suitable for downstream affect-to-bias mapping.

## Miscellaneous search strategy details

During concept extraction, the following will be tracked separately:

* canonical bias names
* alternative labels
* umbrella concepts
* subtype concepts
* mechanism labels
* overlapping or disputed constructs
* historical names no longer preferred

This separation is intended to prevent inflation of the final taxonomy through synonym duplication.

---

# Screening

## Screening stages

1. Title and abstract screening
2. Full-text eligibility assessment
3. Concept-level screening for extractability
4. Redundancy screening during normalization
5. Cyber-relevance screening
6. Final taxonomic inclusion screening

## Deduplication

Deduplication will be performed first through reference-management tools and then through manual review of near-duplicates, expanded conference-to-journal versions, and alternative records of the same work.

## Reliability and reconciliation

Primary screening will be conducted by one reviewer. To improve reliability, a delayed duplicate screening procedure will be applied to a calibration subset of records. Borderline records will default to full-text review rather than early exclusion.

## Justification

The review is theory-building and taxonomy-oriented. At this stage, comprehensive concept recovery is more important than aggressive early exclusion. The screening procedure is therefore designed to minimize concept loss.

## Screened fields / blinding

The following fields may be reviewed during screening:

* title
* abstract
* keywords
* source type
* year
* journal or venue
* domain relevance indicators

Reviewer blinding to author identity and venue is not feasible in practice and will not be attempted.

## Used exclusion criteria

* not relevant to human cognitive bias
* insufficient conceptual or definitional content
* duplicate or superseded version
* no extractable bias construct
* purely technical or nonhuman bias use
* too peripheral to support taxonomy construction

## Screener instructions

When uncertainty is substantive, include for full-text review. Exclude at title/abstract stage only when non-relevance is clear.

## Screening reliability

* Delayed duplicate screening on a calibration subset
* Stability check of included concept set after full-text screening
* Logging of all exclusion reasons at full-text stage

## Screening reconciliation procedure

Where inconsistent decisions occur during delayed duplicate screening, the more inclusive decision will be retained unless a formal exclusion reason is documented at full-text stage.

## Sampling and sample size

No sampling will be applied to eligible records. All records meeting inclusion criteria within the defined search window will be considered.

## Screening procedure justification

Bias terminology is historically layered and discipline-dependent. Overly narrow screening would likely distort taxonomy coverage, collapse meaningful distinctions, or omit important conceptual variants.

## Data management and sharing

The screening log, deduplicated bibliography, and exclusion-reason file will be deposited on OSF, subject to copyright restrictions on full texts.

## Miscellaneous screening details

### Cyber-relevance tagging rule

A concept will enter the final taxonomy only if it is judged relevant to at least one cyber-relevant decision function, such as:

* trust assessment
* credibility assessment
* urgency handling
* authority response
* disclosure or verification decisions
* exception handling
* risk evaluation under uncertainty
* compliance or policy adherence
* installation/update/action initiation decisions
* escalation, reporting, or ignoring

Each leaf node will receive one of two evidence tags:

* **Direct cyber relevance**: explicitly supported in cyber, phishing, social-engineering, misinformation, or security-adjacent literature
* **Structural cyber relevance**: not directly studied in cyber literature but theoretically mappable to one or more predefined cyber-relevant decision functions

This rule is intended to keep the taxonomy scoped to the actual study domain rather than to all named biases ever proposed.

---

# Extraction

## Entities to extract

### Source-level entities

* bibliographic metadata
* source type
* discipline/domain
* named cognitive bias terms
* alternative labels and synonyms
* explicit definitions
* mechanism descriptions
* examples or paradigmatic cases
* stated distinctions from related biases
* parent/child or family relations if given
* notes on ambiguity, overlap, or redundancy
* evidential role of the source

### Concept-level entities

* provisional canonical label
* synonym set
* definitional core
* candidate parent family
* candidate sibling group
* candidate child nodes
* leaf candidacy
* scope note
* confidence tier
* cyber relevance tag
* cyber relevance evidence tier

## Extraction stages

1. Literal term extraction
2. Definition extraction
3. Concept normalization
4. Synonym clustering
5. Concept-splitting and concept-merging decisions
6. Cyber-relevance assignment
7. Hierarchical placement
8. Leaf-node determination
9. Taxonomy freeze

## Extractor instructions

Bias terms should first be extracted exactly as written. Merging should not occur during initial extraction. Definitional nuance should be preserved. Ambiguous cases should be marked explicitly for later adjudication.

## Extractor masking

Not applicable.

## Extraction reliability

A delayed re-extraction will be conducted on a subset of included sources and on a subset of normalized concepts to assess intrarater stability.

## Extraction reconciliation procedure

Where re-extraction differs from original extraction, the adjudication log will document the decision, the rationale, and whether the discrepancy affected canonical labeling, hierarchy placement, leaf status, or cyber-relevance assignment.

## Extraction procedure justification

The main validity threat in taxonomy construction is premature concept collapse. A staged extraction workflow reduces the risk of merging distinct constructs too early.

## Data management and sharing

Extraction sheets, canonicalization logs, synonym maps, cyber-relevance maps, and hierarchy decision logs will be archived on OSF.

## Miscellaneous extraction details

LLMs may be used in a strictly assistive role to suggest candidate synonym sets or summarize definitional contrasts, but final extraction and taxonomy decisions will remain human-authored and explicitly logged.

---

# Synthesis and Quality Assessment

## Planned data transformations

1. Canonicalization of extracted bias terms into a non-redundant concept inventory
2. Recursive hierarchical placement of concepts into family, subfamily, and leaf levels
3. Freezing of the final author-curated 200+ emotion lexicon
4. Standardization of emotion embeddings or vectors prior to clustering
5. Consensus clustering of emotions into higher-order affective clusters
6. Estimation of latent-dimension coordinates for each emotion term
7. Construction of the full emotion × leaf-bias scoring matrix
8. Aggregation of judge outputs across model families, prompt variants, scenarios, and personas
9. Construction of weighted bipartite and projected networks
10. Dimensionality reduction for visualization only, not for primary inference

## Missing data

Invalid, malformed, or abstaining model outputs will receive one controlled retry under a paraphrased but semantically equivalent prompt. Persistent failures will be coded as missing. Missingness will not be imputed in the primary analyses unless a preregistered missingness threshold is exceeded, in which case the affected cell will be flagged as under-supported.

## Data validation

* output-schema validation
* score-range validation
* duplicate-run checks
* prompt-order checks
* contradiction checks across paraphrases
* neutral-baseline sanity checks
* judge-family divergence checks
* scenario and persona perturbation checks
* dimension-anchor checks
* calibration checks against overlapping word norms where available

## Quality assessment

### Taxonomy confidence tier

Each leaf node will be assigned a taxonomy confidence tier based on:

* definitional clarity
* independence of supporting sources
* stability of hierarchical placement
* redundancy risk
* conceptual separability from neighboring nodes
* cyber relevance clarity
* evidence-tier strength

### Emotion-dimension reliability tier

Each emotion’s V/A/C/U/S profile will receive a reliability tier based on:

* cross-model convergence
* prompt stability
* agreement with anchor definitions
* agreement with overlapping lexical norms where available
* contradiction flags

### Simulation reliability tier

Each scored cell will be assigned a reliability tier based on:

* cross-model directional agreement
* prompt stability
* persona stability
* scenario stability
* contradiction flags
* judge confidence profile

---

## Synthesis plan

### Part A: Hierarchical taxonomy construction

The taxonomy will be built recursively using a mechanism-first logic and may include top-level families such as:

* attention and salience distortions
* evidence search and belief-updating distortions
* interpretation and attribution distortions
* memory and retrieval distortions
* uncertainty, probability, and risk distortions
* time, reward, and intertemporal distortions
* social influence and interpersonal judgment distortions
* self-evaluation and ego-protective distortions
* action, persistence, and commitment distortions
* moral, group, and norm-related distortions

A concept will qualify as a leaf node only if it is sufficiently specific, conceptually separable, interpretable on its own, cyber-relevant under the preregistered rule, and useful as a direct scoring target.

### Part B: Object of inference for the scoring phase

The object of inference is:

> the expected direction and magnitude of change in susceptibility to a specific leaf-level cognitive bias, conditional on a transient emotional state, relative to a neutral affect reference condition, within a standardized cyber-relevant decision context.

This is a statement about modeled susceptibility shift, not about deterministic behavior and not about live human targeting.

### Part C: LLM-judge scoring framework

Each scoring prompt will specify:

1. the focal leaf-level bias
2. a standardized cyber-relevant scenario
3. a synthetic persona profile
4. the target emotion condition
5. the neutral reference condition
6. scoring instructions
7. confidence instructions
8. a brief rationale requirement

Judges will provide:

* a direction decision
* a magnitude judgment
* a signed final score from −1000 to +1000
* a confidence estimate
* a short analytic justification

To reduce anchoring and false precision, the rating protocol will use anchored bands while still allowing one-unit increments. The primary confirmatory scale interpretation will rely on magnitude bands and normalized model-aggregated estimates rather than raw single-model integers alone.

### Part D: Judge ensemble and calibration

At least three distinct judge model families or versions will be used where feasible. No primary conclusion will depend on a single model family. This is deliberate because recent LLM-as-a-judge work explicitly identifies reliability, standardization, and bias as major challenges, and empirical work shows that both human and LLM judges can exhibit nontrivial judgment biases and perturbation sensitivity. ([ScienceDirect][8])

To address likely reviewer criticism, the study will pre-register the following safeguards:

* prompt-template replication
* prompt-order randomization
* blinded scoring relative to nonessential source metadata
* model-family triangulation
* anchor-based calibration
* pairwise-comparison calibration subset
* normalization of judge-specific scale usage before aggregation
* instability flags rather than forced consensus

### Part E: Scenario design

The primary scenario families will be cyber-relevant but non-operational and non-instructional. Planned families include:

* trust decisions involving messages, requests, or prompts
* urgency-driven authorization or approval decisions
* requests for disclosure, verification, or exception handling
* installation, update, or access-related judgment decisions
* internal or organizational decision situations involving authority, ambiguity, or perceived consequences
* misinformation or credibility assessment contexts

Scenarios will be abstract enough to avoid functioning as attack templates while still preserving psychologically meaningful decision structure.

### Part F: Synthetic persona design

Synthetic personas will vary along dimensions such as:

* digital literacy
* organizational role seniority
* workload or cognitive load
* security familiarity
* uncertainty tolerance
* baseline trust orientation
* fatigue or depletion indicators

The goal is not demographic realism per se, but structured variation in psychologically relevant moderator dimensions.

### Part G: Emotion clustering

The author-curated 200+ emotion lexicon will be clustered using the supplied emotion vectors or equivalent embedding representation. Cluster estimation will be data-driven rather than imposed a priori. UMAP will be used for two-dimensional visualization and figure production only; it is a standard scalable manifold-learning method commonly used for high-dimensional visualization and is specifically suitable here as an exploratory view rather than as a primary inferential tool. ([arXiv][9])

### Part H: Latent-dimension analyses

In addition to semantic clustering, the study will estimate continuous V/A/C/U/S coordinates for every emotion and use those coordinates in confirmatory models.

The latent-dimension module will address the following reviewer-sensitive question:

> Are the observed bias-vulnerability patterns tied only to specific emotion words, or do they generalize across broader affective and appraisal dimensions?

Planned confirmatory analyses include:

* regression of leaf-level susceptibility on V/A/C/U/S coordinates
* family-level mixed-effects models using latent coordinates as predictors
* interaction models among selected dimensions, especially U × C and V × S
* comparison of discrete-label, cluster-level, and latent-dimensional explanatory power
* held-out emotion prediction analyses to test whether dimensional models generalize beyond lexical labels

This layer is motivated by evidence that emotional space is richer than two dimensions alone, that appraisal dimensions such as certainty and control meaningfully differentiate emotions, and that emotional categories are linked by gradients rather than isolated bins. ([jstor.org][10])

### Part I: Naturalistic composite-emotion module

Because cyber-relevant decision states are rarely emotionally pure, the study will include a preregistered module of naturalistic multi-component emotional states.

The composite module will include a fixed set of preregistered dyads and triads chosen according to:

* appraisal compatibility
* real-world plausibility
* theoretical relevance to pressured decision contexts
* within-cluster and cross-cluster coverage
* manageable total design size
* interpretability of the resulting state

Composite classes will include:

* congruent threat states
* affiliation-pressure states
* depletion-overload states
* certainty-dominance states
* hope-pressure states
* moral-self-evaluative states
* suspicion-hostility states

Illustrative examples include:

* hope + urgency
* guilt + empathy
* fear + confusion
* suspicion + anger
* relief + gratitude
* exhaustion + overwhelm
* pride + certainty
* sadness + loneliness
* optimism + time pressure
* shame + dependency

The final composite set will be frozen before scoring and archived as **Appendix B**.

### Part J: Confirmatory analyses

The confirmatory analyses will include:

* the full emotion × leaf-bias susceptibility matrix
* leaf-level and family-level summary estimates
* cluster-level contrasts
* latent-dimension mixed-effects models
* comparison of discrete-label, cluster, and latent-dimension models
* projected emotion–emotion similarity analysis based on shared bias profile
* projected bias–bias similarity analysis based on shared emotional profile
* non-additivity tests for composite emotions
* uncertainty intervals via bootstrap resampling
* permutation-based robustness checks for network structure
* held-out-emotion generalization tests

### Part K: Exploratory analyses

Exploratory analyses may include:

* alternative cluster-number solutions
* centrality-based identification of especially affect-sensitive bias nodes
* identification of “hub emotions” associated with broad vulnerability profiles
* comparison of different taxonomy granularities
* exploratory canonical-correlation or multiblock analyses linking emotion dimensions and bias families
* exploratory manifold alignment between emotion space and bias-vulnerability space

---

## Criteria for conclusions / inference criteria

### Cell-level association

A cell-level effect will be treated as robust when all of the following are met:

* aggregated signed score departs meaningfully from zero
* directional agreement across judge families is high
* prompt perturbations do not materially reverse the sign
* scenario and persona perturbations do not collapse the effect
* no severe contradiction flags remain unresolved

### Cluster-level association

A cluster-level finding will be treated as robust when:

* multiple constituent emotions show convergent directional effects
* multiple related bias leaves show coherent patterning
* projected-network structure remains stable under sensitivity checks
* the pattern survives removal of any single judge family

### Latent-dimension association

A dimension-level finding will be treated as robust when:

* coefficient direction is stable across model families
* effects remain after perturbation of scenario and persona structure
* predictive performance exceeds chance in held-out-emotion tests
* the result is not driven entirely by one semantic cluster or one narrow subset of labels

---

## Synthesist blinding

Blinding is not feasible for conceptual synthesis. However, source metadata irrelevant to the scoring task will not be presented to judge models.

## Synthesis reliability

Reliability will be evaluated through convergence across judge families, prompt variants, scenarios, persona profiles, and emotion-representation views rather than through single-pass outputs.

## Synthesis reconciliation procedure

Strong disagreement across judge families or perturbation conditions will not be forced into consensus. Instead, those cells will be flagged as unstable, and conclusions will be restricted accordingly.

## Publication bias analyses

Formal publication-bias analysis is not the central issue for this design. Instead, the synthesis will assess:

* dependence on a small number of highly cited conceptual sources
* sensitivity to exclusion of grey literature
* sensitivity to exclusion of domain-specific bounded reviews
* historical concept-saturation patterns

## Sensitivity analyses / robustness checks

* more aggressive synonym collapse
* more conservative synonym separation
* alternative hierarchical depth cutoffs
* alternative emotion-cluster solutions
* removal of one judge family at a time
* removal of one scenario family at a time
* removal of one persona family at a time
* direct-scoring versus pairwise-ranking prompt formats on a subset
* single-emotion-only versus composite-inclusive analyses
* exclusion of low-confidence leaf nodes
* exclusion of low-reliability dimension profiles
* discrete-label-only versus multi-view-representation comparisons

## Synthesis procedure justification

The scientific problem is structural and relational. The study is designed to identify organized mappings between a large affective state space and a hierarchical cognitive-bias space, and to do so under conditions where direct large-scale human data collection would be costlier, less privacy-preserving, and harder to justify at this exploratory mapping stage.

## Synthesis data management and sharing

The OSF project will archive:

* the frozen taxonomy
* synonym and adjudication logs
* the 200+ emotion lexicon
* the composite-emotion set
* emotion-dimension anchor definitions
* scenario templates
* persona templates
* scoring prompts and prompt variants
* raw and processed scoring outputs, subject to model-provider terms
* analysis scripts
* visualization specifications

## Miscellaneous synthesis details

This study is simulation-first, non-operational, and non-deployment-oriented. It is not designed to generate real-world persuasion scripts, targeting procedures, or attack workflows.

---

# Metadata

## Contributors

**Stijn Van Severen**

## Description

This preregistration describes a rapid review and simulation-based study that will construct a comprehensive hierarchical taxonomy of cognitive biases relevant to cyber-relevant decision contexts and map a fixed author-curated 200+ emotion lexicon onto leaf-level bias susceptibility using LLM ensemble judging, semantic clustering, latent-dimension modeling, composite-emotion analysis, and network analysis.

## Registration Type

Public OSF preregistration

## Registry

OSF Registries

## Associated Project

**Cognitive Bias Vulnerability Across Emotional States in Cyber-relevant Decision Contexts**

## Date Created

2026-04-08

## Date Registered

Date of formal OSF submission

## License

MIT License

## Internet Archive Link

To be generated by OSF after registration

## Registration DOI

To be assigned after public registration, if applicable

## Affiliated Institutions

**Universiteit Gent**

## Subjects

* Cybersecurity
* Cyberpsychology
* Cognitive Science
* Judgment and Decision-Making
* Human Factors
* Artificial Intelligence
* Computational Social Science

## Tags

* cognitive bias
* emotion
* affect
* cyberpsychology
* cybersecurity
* social engineering
* human vulnerability
* LLM-as-a-judge
* rapid review
* taxonomy
* UMAP
* network analysis
* OSF preregistration

---

# OSF “Create Project” fields

**Title**
Cognitive Bias Vulnerability Across Emotional States in Cyber-Relevant Decision Contexts

**Storage Location**
Germany – Frankfurt. OSF support documentation states that Germany–Frankfurt is an available storage location and that storage location can be set for newly created projects and on a per-project basis upon creation. ([OSF Support][11])

**Affiliation**
Universiteit Gent

**Description**
Preregistered rapid review and simulation-based study examining how a fixed author-curated 200+ emotion lexicon maps onto a hierarchical taxonomy of cognitive-bias susceptibility in cyber-relevant decision contexts. The project combines taxonomy construction, semantic clustering, latent-dimension modeling, LLM ensemble judging, and network analysis to produce a structured map of cognitive bias vulnerability across emotional states.

**Template**
Blank/general OSF project, followed by creation of the registration from the project. OSF support documentation states that registrations can be started from scratch or from an existing project. ([OSF Support][1])

---

# Governance / ethics wording block

This study is a simulation-first investigation of cognitive vulnerability in cyber-relevant decision contexts. It does not involve live persuasive targeting, operational phishing, real-time profiling of natural persons, or deployment of AI systems for emotion inference on human subjects. The core study relies on literature-derived constructs, synthetic scenarios, synthetic personas, and model-based structured judgments. This substantially limits the need to collect personal data and is consistent with the GDPR’s general data-protection framework. It also fits more comfortably within the EU’s current risk-based AI governance environment than any live manipulative design would. Any later extension involving human participants would require separate ethical review, explicit lawful-basis analysis, and an updated data-governance plan before implementation. ([EUR-Lex][4])

---

# One-sentence abstract version

This preregistered study will construct a comprehensive hierarchical taxonomy of cognitive biases relevant to cyber-relevant decision contexts and estimate how a fixed author-curated lexicon of 200+ emotional states and naturalistic emotion composites shifts susceptibility to leaf-level biases using structured LLM ensemble judging, semantic clustering, latent-dimension modeling, and network analysis.

[1]: https://help.osf.io/article/330-welcome-to-registrations "Welcome to Registrations & Preregistrations! - OSF Support"
[2]: https://www.nist.gov/cyberframework "Cybersecurity Framework | NIST"
[3]: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.01755/pdf "Human Cognition Through the Lens of Social Engineering Cyberattacks"
[4]: https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng?utm_source=chatgpt.com "Regulation - 2016/679 - EN - gdpr - EUR-Lex"
[5]: https://pdodds.w3.uvm.edu/research/papers/others/1980/russell1980a.pdf?utm_source=chatgpt.com "A Circumplex Model of Affect"
[6]: https://link.springer.com/article/10.3758/s13428-012-0314-x "Norms of valence, arousal, and dominance for 13,915 English lemmas | Behavior Research Methods | Springer Nature Link"
[7]: https://www.jstor.org/stable/pdf/26487986.pdf?utm_source=chatgpt.com "Self-report captures 27 distinct categories of emotion bridged by ..."
[8]: https://www.sciencedirect.com/science/article/pii/S2666675825004564 "A survey on LLM-as-a-Judge - ScienceDirect"
[9]: https://arxiv.org/abs/1802.03426 "[1802.03426] UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"
[10]: https://www.jstor.org/stable/40064702?utm_source=chatgpt.com "The World of Emotions Is Not Two-Dimensional - JSTOR"
[11]: https://help.osf.io/article/236-set-a-global-storage-location "Set a Global Storage Location - OSF Support"
