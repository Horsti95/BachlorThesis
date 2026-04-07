# AI-Assisted Development Documentation

**Project**: Intelligent Fingerprint-Based Caching for Iterative ML Experiments  
**Author**: Lennart Gorzel  
**Tool Used**: Claude (Anthropic), used as a development assistant  

---

## How AI Was Used

AI was used as a **coding assistant** throughout the development process — similar to pair programming with an experienced developer. All architectural decisions, experimental design choices, and scientific methodology were planned and directed by the author. AI helped accelerate implementation, catch bugs, and improve code quality.

**AI was NOT used for:**
- Writing the thesis text
- Designing the experimental methodology
- Choosing the caching strategy or fingerprint approach
- Interpreting results or drawing conclusions

**AI WAS used for:**
- Code implementation based on my specifications
- Code review and bug detection
- Refactoring and cleanup
- Debugging runtime errors

---

## Example Prompts (Selected)

### Architecture & Design

> "I need a two-layer caching system: Layer 1 caches extracted features per subject using SHA-256 fingerprints. Layer 2 caches trained LOSO fold models. The fingerprint must include the held-out subject to prevent data leakage. Can you implement this in fingerprint.py?"

> "Walk me through how the feature selection pipeline should work. I want correlation filter first, then ANOVA top-K. The scope should be configurable: global fits once on all data for speed, per-fold fits on training data only for methodological purity."

> "Design the training grid for my thesis: 2 models (XGBoost, Random Forest) x 3 correlation thresholds (0.75, 0.90, None) x 3 top-K values (30, 50, None) = 18 configurations. Each runs 128 LOSO folds."

### Implementation

> "Implement the LOSO model cache in loso_cache.py. It should generate a fingerprint from the model config, feature selection params, and held-out subject ID. Cache hit = load model from disk and skip training. Cache miss = train, save, continue."

> "Add a global feature selection path in training.py. When scope is 'global', fit the feature selection pipeline once on all data, then reuse the selected features for all 128 folds. This avoids refitting 128 times."

> "Create the output formatter. I want box-style headers for each config, compact fold progress lines, and an aggregate results table at the end. Support quiet/normal/verbose modes."

### Bug Fixing & Debugging

> "The corrNone_kAll configuration produces NaN results. All other configs work. Debug the training loop — I think the issue is in how we handle the case where both correlation threshold and top-K are None."

> "My pipeline crashes with UnicodeEncodeError when I pipe output through findstr on Windows. The arrow character causes it. Fix all print statements to be Windows-safe."

> "The per-config cache summary says 'time saved: 53min' even on a cold run. That's wrong — it's conflating feature cache hits with model cache hits. Fix it to show actual LOSO model cache statistics."

### Code Review & Cleanup

> "Review the entire codebase for critical bugs. Focus on: data leakage, incorrect metrics, cache invalidation failures, and silent errors. Categorize by severity."

> "The fold training output is too noisy — logger.info from models.py fires 128 times per config and mixes with the fold result lines. Move those to logger.debug so the output is clean."

> "Restructure the pre-run output. Right now it shows the config summary, then cache status, then experiment header, then LOSO setup — all saying overlapping things. Combine into one clean block."

### Benchmarking & Validation

> "I need to verify all 18 configurations produce valid results. Give me a quick command to run 3 subjects across the full thesis grid."

> "The benchmark shows Cold->SSD speedup of 14.6x for XGBoost and 5.4x for RF. RAM cache adds marginal benefit. Document this in the future work notes — thesis focuses on SSD caching."

---

## Development Workflow

1. **I defined the requirement** (e.g., "add model caching with fingerprint invalidation")
2. **AI implemented** based on my specification
3. **I reviewed and tested** on my machine with real data
4. **I reported issues** (e.g., "NaN on config 3, works on others")
5. **AI debugged and fixed** based on my error output
6. **I validated the fix** with another run

This iterative cycle is standard in modern software development with AI tools.
