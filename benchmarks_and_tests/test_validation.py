"""Test the comprehensive data validation."""
from data_loader_boas import BOASDataLoader

# Local BOAS dataset path. Developer-specific — see README "Local data path".
DEFAULT_DATA_PATH = 'C:/Users/DerHo/Desktop/Data'

loader = BOASDataLoader(DEFAULT_DATA_PATH)
raw, ann, meta = loader.load_subject('1', apply_preprocessing=False)

print("\n" + "="*60)
print("COMPREHENSIVE DATA VALIDATION")
print("="*60)

results = loader.validate_subject_data(raw, ann, meta, check_signal_integrity=True)

print("\nValidation Results:")
for check, passed in results.items():
    if check != '_summary':
        status = '✓' if passed else '✗'
        print(f"  {status} {check}")

summary = results['_summary']
print(f"\nSummary: {summary['passed']}/{summary['total']} checks passed")

if summary['issues']:
    print("\nIssues found:")
    for issue in summary['issues']:
        print(f"  ⚠ {issue}")
else:
    print("\n✓ All data integrity checks passed!")
