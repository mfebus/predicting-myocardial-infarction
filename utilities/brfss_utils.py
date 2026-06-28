"""
brfss_utils.py
==============
Shared BRFSS data utilities for the myocardial-infarction modeling project.

This module owns everything about *reading and labeling* BRFSS data:
constants, the display-label lookups, the humanize() helper, data loading
into a feature/target frame, and the standalone EDA helpers (missingness
and outlier analysis).

Display-only rule: the readable labels here are used for plot labels and
tables ONLY. They are never applied to the model data itself — feature
selection, encoding, and the model all operate on the original BRFSS codes.

Maintained by: Maria
"""

import pickle

import numpy as np
import pandas as pd


# =====================================================================
# CONSTANTS
# =====================================================================

RANDOM_STATE = 42          # fixed seed for reproducible splits, sampling, and models
CV_FOLDS     = 5           # folds for cross-validation
TARGET       = "_MICHD"    # binary target: history of MI / coronary heart disease


# =====================================================================
# DISPLAY LABELS  (display-only — never applied to model data)
# =====================================================================

# Maps each of the 27 selected feature codes to a readable label.
# Definitions follow the BRFSS 2024 codebook (USCODE24_LLCP_082125.HTML).
LABELS = {
    "_AGEG5YR": "Age group",
    "_RFHLTH":  "General health",
    "_INCOMG1": "Income group",
    "MARITAL":  "Marital status",
    "PRIMINS2": "Primary insurance",        # note: Medicare acts as an age proxy
    "VETERAN3": "Veteran",
    "PREGNANT": "Pregnant",
    "_SMOKER3": "Smoking status",
    "_LCSYSMK": "Years smoked",
    "LCSNUMCG": "Cigarettes per day",
    "_LCSYQTS": "Years since quitting",
    "DIABETE4": "Diabetes",
    "CVDSTRK3": "Stroke history",           # most irreplaceable feature by ablation
    "CHCCOPD3": "COPD",
    "CHCKDNY2": "Kidney disease",
    "HAVARTH4": "Arthritis",
    "CHCOCNC1": "Cancer history",
    "DIFFWALK": "Difficulty walking",
    "DIFFDRES": "Difficulty dressing",
    "DIFFALON": "Difficulty doing errands alone",
    "DEAF":     "Hearing difficulty",
    "PHYSHLTH": "Physically unhealthy days",   # continuous
    "POORHLTH": "Poor-health days",            # continuous
    "CHECKUP1": "Last checkup",
    "PERSDOC3": "Has personal doctor",
    "PNEUVAC4": "Pneumonia vaccine",
    "_TOTINDA": "Physical activity",           # also the uplift treatment variable
}

# BRFSS _AGEG5YR code -> readable age range (display-only)
AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+", 14: "Unknown",
}


def humanize(name):
    """Convert a BRFSS feature code into a human-readable label.

    Handles two cases:
      1. Exact match -> "_AGEG5YR"                  -> "Age group"
      2. One-hot column from the ColumnTransformer, where the encoder
         appends the category value after an underscore:
                      -> "_RFHLTH_Fair or Poor Health"
                      -> "General health = Fair or Poor Health"

    Codes are checked longest-first so a longer, more specific code is
    matched before a shorter code that is a prefix of it. Any name with
    no matching code is returned unchanged, so this is safe to map over
    a full list of (possibly already-clean) column names.

    Note: strip the ColumnTransformer lane prefix ("categorical__" /
    "continuous__") BEFORE calling, or the lookup will not match.
    """
    # Longest codes first -> avoids a short code matching a longer code's column
    for code in sorted(LABELS, key=len, reverse=True):
        if name == code:                       # exact feature match
            return LABELS[code]
        if name.startswith(code + "_"):        # one-hot column: "CODE_<value>"
            return f"{LABELS[code]} = {name[len(code) + 1:]}"
    return name                                # unknown -> leave unchanged


# =====================================================================
# DATA LOADING
# =====================================================================

def load_data(data_path, features_path):
    """Load the processed BRFSS dataset and the selected-feature list.

    Builds the categorical/continuous column lists used by the modeling
    pipeline (only _MICHD, the target, is binary), and
    drops rows with a missing target.

    Parameters
    ----------
    data_path : str
        Path to the processed CSV (e.g. processed_data_v2.csv on Drive).
    features_path : str
        Path to the selected-features pickle (e.g. selected_features_v2.pkl).

    Returns
    -------
    dict with keys:
        df               : the loaded DataFrame (rows with NaN target dropped)
        FEATURES         : the 27 selected features that survived selection
        categorical_cols : all categorical columns in df (full dataset)
        continuous_cols  : all continuous columns in df (full dataset)
        cat_features     : selected features that are categorical
        cont_features    : selected features that are continuous
    """
    # --- Load the processed dataset (cleaning/selection done upstream) ---
    df = pd.read_csv(data_path, low_memory=False)

    # --- Load the selected features (two-stage significance +
    #     multicollinearity selection from the EDA notebook) -----------
    with open(features_path, "rb") as f:
        selected_features = pickle.load(f)
    FEATURES = list(selected_features)   # the 27 features that survived selection

    # --- Build the two feature-type lanes for the ColumnTransformer ------
    # Logic mirrors the EDA notebook
    # Categorical: object/category dtype OR low-cardinality numeric (<10 unique).
    categorical_cols = [
        c for c in df.columns
        if c != TARGET
        and (df[c].dtype == "object" or df[c].dtype.name == "category" or df[c].nunique() < 10)
    ]
    # Continuous: remaining numeric columns that aren't categorical.
    continuous_cols = [
        c for c in df.select_dtypes("number").columns
        if c != TARGET
        and c not in categorical_cols
    ]

    # --- Drop rows with a missing target (unusable for supervised learning) ---
    df = df[df[TARGET].notna()]

    # --- Separate the SELECTED features by type (excludes target) --------
    cat_features  = [f for f in FEATURES if f in categorical_cols]
    cont_features = [f for f in FEATURES if f in continuous_cols]

    return {
        "df": df,
        "FEATURES": FEATURES,
        "categorical_cols": categorical_cols,
        "continuous_cols": continuous_cols,
        "cat_features": cat_features,
        "cont_features": cont_features,
    }


# =====================================================================
# EDA HELPERS
# =====================================================================

def find_missingness_drivers(df, missing_threshold=0.10, top_n=3, min_group=200):
    """For each high-missing feature, find which column best explains its missingness.

    Determines whether missingness is random or structural by checking, for
    each feature missing >= `missing_threshold`, which other column most
    sharply splits its missing/not-missing pattern.
    """
    # Fraction of rows missing in each column (True=1/False=0, so .mean() = % missing)
    miss_rate = df.isna().mean()

    # Keep only features missing at least `missing_threshold` — worth investigating
    high_missing = miss_rate[miss_rate >= missing_threshold].index.tolist()

    summary = []
    for feature in high_missing:

        is_missing = df[feature].isna()

        candidates = []
        for col in df.columns:
            if col == feature:          # don't test a feature against itself
                continue

            # Split the missingness flag into groups defined by the candidate driver
            grp = is_missing.groupby(df[col])

            rates  = grp.mean()         # within each group, what % is missing
            counts = grp.size()         # how many people are in each group

            # Drop tiny groups — a 100% rate from 3 people is noise, not signal
            rates = rates[counts >= min_group]
            if len(rates) < 2:          # need at least 2 groups to compare
                continue

            # The key metric: how far apart are the highest and lowest missing-rates?
            # Big spread = this column strongly controls whether the feature is missing
            spread = rates.max() - rates.min()
            candidates.append((col, round(spread, 3)))

        # Rank candidates so the strongest driver is first
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:top_n]

        summary.append({
            'feature': feature,
            'pct_missing': round(miss_rate[feature] * 100, 1),
            'top_driver': top[0][0] if top else None,      # best explainer
            'driver_spread': top[0][1] if top else None,   # how clean the split is
            # spread >= 0.7 means missingness flips ~0% to ~100% across groups = skip logic
            'verdict': 'STRUCTURAL' if top and top[0][1] >= 0.7 else 'check',
            'runners_up': top[1:],
        })

    # Show the most-missing features first
    return pd.DataFrame(summary).sort_values('pct_missing', ascending=False)


def outlier_summary(df, cont_features):
    """Flag IQR outliers in each continuous feature.

    A value is an outlier if it falls below Q1 - 1.5*IQR or above
    Q3 + 1.5*IQR. Returns count, percentage (of non-missing), and the
    cutoff bounds per feature, sorted most-outlier-heavy first.
    (Note: tree models like RF are robust to outliers.)
    """
    outlier_rows = []
    for col in cont_features:
        # Quartiles and interquartile range for this feature
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1

        # IQR fences: standard 1.5 * IQR rule
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Flag values outside the fences
        mask  = (df[col] < lower) | (df[col] > upper)
        n_out = mask.sum()

        outlier_rows.append({
            'feature':      col,
            'n_outliers':   n_out,
            # percentage is of NON-missing values, so NaNs don't deflate the rate
            'pct_outliers': round(100 * n_out / df[col].notna().sum(), 1),
            'lower_bound':  round(lower, 2),
            'upper_bound':  round(upper, 2),
        })

    return pd.DataFrame(outlier_rows).sort_values('pct_outliers', ascending=False)

    """
Data Dictionary for the entire/raw BRFSS 2024 dataset
=====================================================
Maps original BRFSS technical column names to short, readable labels.

"""
COLUMN_MAP = {

    # ===== SURVEY METADATA =====
    "_STATE": "state_fips",
    "FMONTH": "file_month",
    "IDATE": "interview_date",
    "IMONTH": "interview_month",
    "IDAY": "interview_day",
    "IYEAR": "interview_year",
    "DISPCODE": "disposition_code",
    "SEQNO": "sequence_number",
    "_PSU": "primary_sampling_unit",
    "QSTVER": "questionnaire_version",
    "QSTLANG": "language",
    "ICFQSTVR": "icf_questionnaire_version",

    # ===== CONTACT / HOUSEHOLD SCREENING =====
    "CTELENM1": "correct_telephone",
    "PVTRESD1": "private_residence",
    "COLGHOUS": "college_housing",
    "STATERE1": "state_resident",
    "CELPHON1": "cellular_phone",
    "LADULT1": "is_adult_18plus",
    "NUMADULT": "num_adults_household",
    "RESPSLC1": "respondent_selection",
    "LANDSEX3": "sex_landline",
    "SAFETIME": "safe_time_to_talk",
    "CTELNUM1": "correct_phone_number",
    "CELLFON5": "is_cell_phone",
    "CADULT1": "is_adult_18plus_cell",
    "CELLSEX3": "sex_cell",
    "PVTRESD3": "private_residence_cell",
    "CCLGHOUS": "college_housing_cell",
    "CSTATE1": "current_state_resident",
    "LANDLINE": "has_landline",
    "HHADULT": "num_adults_household_cell",

    # ===== DEMOGRAPHICS =====
    "SEXVAR": "sex",
    "MARITAL": "marital_status",
    "EDUCA": "education_level",
    "RENTHOM1": "own_or_rent_home",
    "NUMHHOL4": "household_landlines",
    "NUMPHON4": "residential_landlines",
    "CPDEMO1C": "has_personal_cellphone",
    "VETERAN3": "is_veteran",
    "EMPLOY1": "employment_status",
    "CHILDREN": "num_children_household",
    "INCOME3": "income_level",
    "PREGNANT": "pregnancy_status",
    "WEIGHT2": "weight_lbs",
    "HEIGHT3": "height_ft_in",

    # ===== GENERAL HEALTH =====
    "GENHLTH": "general_health",
    "PHYSHLTH": "days_poor_physical_health",
    "MENTHLTH": "days_poor_mental_health",
    "POORHLTH": "days_poor_phys_or_mental",

    # ===== HEALTH CARE ACCESS =====
    "PRIMINS2": "primary_health_coverage",
    "PERSDOC3": "has_personal_doctor",
    "MEDCOST1": "could_not_afford_doctor",
    "CHECKUP1": "time_since_checkup",

    # ===== EXERCISE =====
    "EXERANY2": "exercised_past_30days",

    # ===== ORAL HEALTH =====
    "LASTDEN4": "last_dental_visit",
    "RMVTETH4": "permanent_teeth_removed",

    # ===== CARDIOVASCULAR DISEASE =====
    "CVDINFR4": "ever_heart_attack",
    "CVDCRHD4": "ever_angina_or_chd",
    "CVDSTRK3": "ever_stroke",

    # ===== ASTHMA =====
    "ASTHMA3": "ever_asthma",
    "ASTHNOW": "still_has_asthma",

    # ===== CANCER =====
    "CHCSCNC1": "ever_skin_cancer",
    "CHCOCNC1": "ever_other_cancer",

    # ===== CHRONIC CONDITIONS =====
    "CHCCOPD3": "ever_copd",
    "ADDEPEV3": "ever_depression",
    "CHCKDNY2": "ever_kidney_disease",
    "HAVARTH4": "ever_arthritis",

    # ===== DIABETES =====
    "DIABETE4": "ever_diabetes",
    "DIABAGE4": "age_diabetes_diagnosed",
    "PDIABTS1": "last_blood_sugar_test",
    "PREDIAB2": "ever_prediabetes",
    "DIABTYPE": "diabetes_type",
    "INSULIN1": "takes_insulin",
    "CHKHEMO3": "hemoglobin_checks",
    "EYEEXAM1": "last_eye_exam_dilated",
    "DIABEYE1": "last_diabetic_eye_photo",
    "DIABEDU1": "last_diabetes_education",
    "FEETSORE": "ever_foot_sores",

    # ===== DISABILITY =====
    "DEAF": "is_deaf",
    "BLIND": "is_blind",
    "DECIDE": "difficulty_concentrating",
    "DIFFWALK": "difficulty_walking",
    "DIFFDRES": "difficulty_dressing",
    "DIFFALON": "difficulty_errands_alone",

    # ===== WOMEN'S HEALTH =====
    "HADMAM": "ever_had_mammogram",
    "HOWLONG": "time_since_mammogram",
    "CERVSCRN": "ever_cervical_screening",
    "CRVCLCNC": "time_since_cervical_screen",
    "CRVCLPAP": "had_recent_pap",
    "CRVCLHPV": "had_recent_hpv_test",
    "HADHYST2": "had_hysterectomy",

    # ===== COLORECTAL CANCER SCREENING =====
    "HADSIGM4": "ever_sigmoid_or_colon",
    "COLNSIGM": "ever_colon_or_sigmoid",
    "COLNTES1": "time_since_colonoscopy",
    "SIGMTES1": "time_since_sigmoidoscopy",
    "LASTSIG4": "time_since_sigmoid_colon",
    "COLNCNCR": "other_colorectal_test",
    "VIRCOLO1": "ever_virtual_colonoscopy",
    "VCLNTES2": "time_since_virtual_colon",
    "SMALSTOL": "ever_stool_test",
    "STOLTEST": "time_since_stool_test",
    "STOOLDN2": "ever_stool_dna_test",
    "BLDSTFIT": "was_cologuard_test",
    "SDNATES1": "time_since_stool_dna",

    # ===== TOBACCO USE =====
    "SMOKE100": "smoked_100_cigarettes",
    "SMOKDAY2": "current_smoking_frequency",
    "USENOW3": "smokeless_tobacco_use",
    "ECIGNOW3": "ecigarette_use",
    "LCSFIRST": "age_started_smoking",
    "LCSNUMCG": "cigarettes_per_day",
    "LASTSMK2": "time_since_last_smoke",
    "STOPSMK2": "stopped_smoking_12mo",
    "MENTCIGS": "uses_menthol_cigarettes",
    "MENTECIG": "uses_menthol_ecigs",
    "HEATTBCO": "heard_of_heated_tobacco",

    # ===== LUNG CANCER SCREENING =====
    "LCSCTSC1": "had_ct_scan",
    "LCSSCNCR": "ct_for_lung_cancer",
    "LCSCTWHN": "time_since_ct_scan",

    # ===== ALCOHOL =====
    "ALCDAY4": "alcohol_days_past_30",
    "AVEDRNK4": "avg_drinks_per_day",
    "DRNK3GE5": "binge_drinking",
    "MAXDRNKS": "max_drinks_one_occasion",

    # ===== IMMUNIZATIONS =====
    "FLUSHOT7": "flu_shot_past_year",
    "FLSHTMY3": "last_flu_shot_date",
    "IMFVPLA5": "flu_shot_location",
    "PNEUVAC4": "ever_pneumonia_shot",
    "SHINGLE2": "ever_shingles_vaccine",
    "HPVADVC4": "ever_hpv_vaccine",
    "HPVADSH1": "num_hpv_shots",
    "HPVDSHT": "num_hpv_shots_v2",
    "TETANUS1": "tetanus_shot_since_2005",

    # ===== HIV =====
    "HIVTST7": "ever_hiv_tested",
    "HIVTSTD3": "last_hiv_test_date",
    "HIVRISK5": "high_risk_situations",

    # ===== ARTHRITIS MANAGEMENT =====
    "ARTHEXER": "doctor_suggested_exercise",

    # ===== CANCER SURVIVORSHIP =====
    "CNCRDIFF": "num_cancer_types",
    "CNCRAGE": "age_cancer_diagnosed",
    "CNCRTYP2": "cancer_type",
    "CSRVTRT3": "current_cancer_treatment",
    "CSRVDOC1": "cancer_primary_doctor",
    "CSRVSUM": "received_treatment_summary",
    "CSRVRTRN": "received_followup_instructions",
    "CSRVINST": "instructions_written",
    "CSRVINSR": "insurance_paid_treatment",
    "CSRVDEIN": "ever_denied_insurance_cancer",
    "CSRVCLIN": "cancer_clinical_trial",
    "CSRVPAIN": "current_cancer_pain",
    "CSRVCTL2": "pain_under_control",

    # ===== PROSTATE CANCER SCREENING =====
    "PSATEST1": "ever_psa_test",
    "PSATIME1": "time_since_psa",
    "PCPSARS2": "psa_main_reason",
    "PSASUGS1": "psa_first_suggested_by",
    "PCSTALK2": "discussed_psa_pros_cons",

    # ===== COGNITIVE DECLINE =====
    "CIMEMLO1": "cognitive_difficulties",
    "CDWORRY": "worried_about_cognition",
    "CDDISCU1": "discussed_cognition_provider",
    "CDHOUS1": "given_up_chores_cognition",
    "CDSOCIA1": "cognition_affects_work_social",

    # ===== CAREGIVING =====
    "CAREGIV1": "is_caregiver",
    "CRGVREL5": "caregiving_relationship",
    "CRGVPRB4": "caregiving_main_problem",
    "CRGVALZD": "care_recipient_alzheimers",
    "CRGVNURS": "caregiving_nursing_tasks",
    "CRGVPER2": "caregiving_personal_care",
    "CRGVHOU2": "caregiving_household_tasks",
    "CRGVHRS2": "caregiving_hours",
    "CRGVLNG2": "caregiving_duration",

    # ===== ADVERSE CHILDHOOD EXPERIENCES (ACE) =====
    "ACEDEPRS": "ace_lived_with_depressed",
    "ACEDRINK": "ace_lived_with_drinker",
    "ACEDRUGS": "ace_lived_with_drug_user",
    "ACEPRISN": "ace_lived_with_incarcerated",
    "ACEDIVRC": "ace_parents_divorced",
    "ACEPUNCH": "ace_parents_violent",
    "ACEHURT1": "ace_parent_hurt_you",
    "ACESWEAR": "ace_parent_swore_at_you",
    "ACETOUCH": "ace_sexual_touch",
    "ACETTHEM": "ace_forced_touch_them",
    "ACEHVSEX": "ace_forced_sex",
    "ACEADSAF": "ace_adult_made_you_safe",
    "ACEADNED": "ace_basic_needs_met",

    # ===== SOCIAL DETERMINANTS OF HEALTH =====
    "LSATISFY": "life_satisfaction",
    "EMTSUPRT": "emotional_support",
    "SDLONELY": "loneliness_frequency",
    "SDHEMPLY": "lost_employment_or_hours",
    "FOODSTMP": "received_food_stamps",
    "SDHFOOD1": "food_insecurity",
    "SDHBILLS": "unable_to_pay_bills",
    "SDHUTILS": "unable_to_pay_utilities",
    "SDHTRNSP": "lack_of_transportation",
    "HOWSAFE1": "neighborhood_safety",

    # ===== MARIJUANA =====
    "MARIJAN1": "marijuana_days_past_30",
    "USEMRJN4": "used_marijuana",
    "MARJSMOK": "smoked_marijuana",
    "MARJEAT": "ate_marijuana",
    "MARJVAPE": "vaped_marijuana",
    "MARJDAB": "dabbed_marijuana",
    "MARJOTHR": "other_marijuana_use",

    # ===== SUGAR-SWEETENED BEVERAGES =====
    "SSBSUGR2": "regular_soda_frequency",
    "SSBFRUT3": "sugar_sweetened_drink_freq",

    # ===== FIREARMS =====
    "FIREARM5": "firearms_in_home",
    "GUNLOAD": "firearms_loaded",
    "LOADULK2": "firearms_loaded_unlocked",

    # ===== CHILDHOOD ASTHMA (parent reporting) =====
    "RCSBORG1": "child_sex",
    "RCSGEND1": "child_gender_v2",
    "RCSXBRTH": "child_sex_at_birth",
    "RCSRLTN2": "relationship_to_child",
    "CASTHDX2": "child_ever_asthma",
    "CASTHNO2": "child_still_asthma",

    # ===== SEXUAL ORIENTATION =====
    "SOMALE": "sexual_orientation_male",
    "SOFEMALE": "sexual_orientation_female",

    # ===== SEXUAL BEHAVIOR / CONTRACEPTION =====
    "HADSEX": "had_sex",
    "PFPPRVN4": "used_contraception",
    "TYPCNTR9": "contraception_type",
    "NOBCUSE8": "no_contraception_reason",

    # ===== GEOGRAPHIC / SURVEY DESIGN =====
    "_METSTAT": "metropolitan_status",
    "_URBSTAT": "urban_rural",
    "MSCODE": "metro_status_code",
    "_STSTR": "stratification_var",
    "_STRWT": "stratum_weight",
    "_RAWRAKE": "raw_weight_factor",
    "_WT2RAKE": "design_weight",
    "_CLLCPWT": "child_final_weight",
    "_DUALUSE": "dual_phone_use",
    "_DUALCOR": "dual_phone_correction",
    "_LLCPWT2": "truncated_design_weight",
    "_LLCPWT": "final_weight",

    # ===== CALCULATED DEMOGRAPHIC VARIABLES =====
    "_IMPRACE": "race_ethnicity_imputed",
    "_CHISPNC": "child_hispanic",
    "_CRACE1": "child_race",
    "CAGEG": "child_age_group",
    "_MRACE1": "race_non_hispanic",
    "_HISPANC": "hispanic_origin",
    "_RACE": "race_ethnicity",
    "_RACEG21": "race_white_vs_other",
    "_RACEGR3": "race_5_level",
    "_RACEPRV": "race_internet_tables",
    "_SEX": "sex_calculated",
    "_AGEG5YR": "age_5yr_groups",
    "_AGE65YR": "age_65_split",
    "_AGE80": "age_capped_80",
    "_AGE_G": "age_6_groups",

    # ===== CALCULATED HEALTH STATUS =====
    "_RFHLTH": "good_health_calc",
    "_PHYS14D": "physical_health_status",
    "_MENT14D": "mental_health_status",
    "_HLTHPL2": "has_health_insurance",
    "_HCVU654": "insurance_18_64",
    "_TOTINDA": "physical_activity_calc",
    "_EXTETH3": "teeth_extracted_18plus",
    "_ALTETH3": "all_teeth_extracted_65plus",
    "_DENVST3": "dental_visit_past_year",
    "_MICHD": "ever_chd_or_mi",
    "_LTASTH1": "lifetime_asthma_calc",
    "_CASTHM1": "current_asthma_calc",
    "_ASTHMS1": "asthma_status",
    "_DRDXAR2": "diagnosed_arthritis",

    # ===== CALCULATED BMI =====
    "HTIN4": "height_inches",
    "HTM4": "height_meters",
    "WTKG3": "weight_kg",
    "_BMI5": "bmi",
    "_BMI5CAT": "bmi_category",
    "_RFBMI5": "overweight_or_obese",

    # ===== CALCULATED FAMILY / EDUCATION / INCOME =====
    "_CHLDCNT": "num_children_calc",
    "_EDUCAG": "education_calc",
    "_INCOMG1": "income_calc",

    # ===== CALCULATED SCREENING VARIABLES =====
    "_RFMAM23": "mammogram_40plus_2yr",
    "_MAM402Y": "mammogram_40_74_2yr",
    "_CRVSCRN": "cervical_screen_calc",
    "_RFPAP37": "pap_test_21_65",
    "_HPV5YR1": "hpv_test_30_65",
    "_PAPHPV1": "pap_or_hpv_test",
    "_HADCOLN": "had_colonoscopy_calc",
    "_CLNSCP2": "colonoscopy_45_75_10yr",
    "_HADSIGM": "had_sigmoid_calc",
    "_SGMSCP2": "sigmoid_45_75_5yr",
    "_SGMS102": "sigmoid_45_75_10yr",
    "_RFBLDS6": "stool_test_45_75",
    "_STOLDN2": "stool_dna_45_75",
    "_VIRCOL2": "virtual_colon_45_75",
    "_SBONTI2": "sigmoid_and_blood_test",
    "_CRCREC3": "met_uspstf_crc_guidelines",

    # ===== CALCULATED SMOKING / LUNG CANCER =====
    "_SMOKER3": "smoking_status",
    "_RFSMOK3": "current_smoker_calc",
    "_CURECI3": "current_ecig_user",
    "LCSLAST_": "age_last_smoked",
    "LCSNUMC_": "cigarettes_per_day_calc",
    "_LCSAGE": "lung_screen_age_groups",
    "_LCSYSMK": "years_smoked",
    "_PACKDAY": "packs_per_day",
    "_PACKYRS": "pack_years",
    "_LCSYQTS": "years_since_quit",
    "_LCSSMKG": "lung_screen_smoking_group",
    "_LCSELIG": "uspstf_screen_eligible",
    "_LCSCTSN": "had_chest_ct_calc",
    "_LCSPSTF": "meets_uspstf_lung_guidelines",

    # ===== CALCULATED ALCOHOL =====
    "DRNKANY6": "drank_any_alcohol_30days",
    "DROCDY4_": "drink_occasions_per_day",
    "_RFBING6": "binge_drinking_calc",
    "_DRNKWK3": "drinks_per_week",
    "_RFDRHV9": "heavy_alcohol_calc",

    # ===== CALCULATED IMMUNIZATIONS =====
    "_FLSHOT7": "flu_shot_calc",
    "_PNEUMO3": "pneumonia_vaccine_calc",
    "_AIDTST4": "ever_hiv_tested_calc",
}


# ==================================================
# Column Renaming Function
# ==================================================

def rename_cols(obj):
    """
    Rename BRFSS field names using the project
    standard COLUMN_MAP dictionary.

    Parameters
    ----------
    obj : pandas.DataFrame, list, or str
        Object containing BRFSS field names.

    Returns
    -------
    pandas.DataFrame, list, or str
        Object of the same type with renamed fields.
    """

    if isinstance(obj, pd.DataFrame):
        return obj.rename(columns=COLUMN_MAP)

    elif isinstance(obj, list):
        return [COLUMN_MAP.get(field, field)
                for field in obj]

    elif isinstance(obj, str):
        return COLUMN_MAP.get(obj, obj)

    else:
        raise TypeError(
            "Input must be a pandas DataFrame, list, or string."
        )


