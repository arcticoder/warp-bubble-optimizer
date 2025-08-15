from tools.validate_profiles import has_distance_column

def test_validate_profile_standard_and_varied():
    assert has_distance_column('data/dist_profile_standard.csv')
    assert has_distance_column('data/dist_profile_varied.csv')
