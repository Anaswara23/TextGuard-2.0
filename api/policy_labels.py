"""
Sector-scoped policy risk labels for TextGuard 2.0.
Each sector has a description (used for auto-detection) and a list of
fine-grained policy labels (used for risk tagging of flagged sections).
"""

POLICY_LABELS_BY_SECTOR = {
    "TRANSPORT": [
        "climate_ghg_emissions",
        "transport_electrification",
        "low_carbon_fuels",
        "transport_demand_management",
        "land_use_and_smart_growth",
        "public_and_active_transportation",
        "transport_infrastructure",
        "climate_resilience_and_adaptation",
        "transport_equity_and_access",
        "transport_safety",
        "freight_and_goods_movement",
        "transport_funding_and_finance",
        "governance_and_regulation",
        "mobility_innovation_and_technology",
        "air_quality_and_health",
        "behavior_change_policies",
        "vehicle_standards_and_efficiency",
        "accessibility_and_disability",
    ],
    "AGRICULTURE_AND_RURAL_DEVELOPMENT": [
        "climate_smart_agriculture",
        "sustainable_land_management",
        "soil_health_and_erosion_control",
        "water_efficiency_and_irrigation",
        "biodiversity_and_agroforestry",
        "livestock_emissions_and_manure",
        "crop_diversification_and_food_security",
        "agricultural_value_chains_and_markets",
        "rural_infrastructure_and_connectivity",
        "land_tenure_and_property_rights",
        "agricultural_extension_and_capacity",
        "gender_and_inclusion_in_agriculture",
        "disaster_risk_management_in_agriculture",
        "pesticide_and_chemical_management",
    ],
    "EDUCATION": [
        "access_to_basic_education",
        "secondary_and_tertiary_education",
        "technical_and_vocational_training",
        "digital_learning_and_infrastructure",
        "teacher_training_and_quality",
        "curriculum_reform_and_standards",
        "inclusive_education_gender_disability",
        "early_childhood_development",
        "education_finance_and_governance",
        "learning_assessment_and_outcomes",
        "school_infrastructure_and_safe_schools",
        "climate_and_environmental_education",
    ],
    "ENERGY": [
        "renewable_energy_generation",
        "energy_efficiency_and_demand_side",
        "grid_modernization_and_reliability",
        "off_grid_and_distributed_energy",
        "fossil_fuel_phaseout_and_transition",
        "energy_access_and_energy_poverty",
        "clean_cooking_and_heating",
        "energy_sector_governance_and_regulation",
        "energy_pricing_and_subsidy_reform",
        "energy_resilience_and_disaster_risk",
    ],
    "ENVIRONMENT_AND_NATURAL_DISASTERS": [
        "climate_mitigation_general",
        "climate_adaptation_general",
        "disaster_risk_reduction",
        "ecosystem_conservation_and_biodiversity",
        "protected_areas_and_land_use_planning",
        "coastal_and_marine_management",
        "pollution_control_and_waste_management",
        "environmental_impact_assessment",
        "environmental_governance_and_institutions",
        "environmental_monitoring_and_data",
    ],
    "FINANCIAL_MARKETS": [
        "financial_sector_regulation_and_supervision",
        "capital_markets_development",
        "banking_sector_resilience",
        "green_bonds_and_sustainable_finance",
        "sme_finance_and_access_to_credit",
        "financial_inclusion_and_digital_finance",
        "microfinance_and_rural_finance",
        "payment_systems_and_financial_infrastructure",
        "anti_money_laundering_and_kyc",
        "consumer_protection_and_financial_literacy",
    ],
    "HEALTH": [
        "primary_health_care_systems",
        "communicable_disease_control",
        "non_communicable_diseases",
        "maternal_child_and_reproductive_health",
        "health_emergency_preparedness",
        "health_infrastructure_and_equipment",
        "health_workforce_and_training",
        "health_financing_and_insurance",
        "digital_health_and_information_systems",
        "environmental_health_and_sanitation",
        "mental_health_and_psychosocial_support",
        "health_equity_gender_and_vulnerable_groups",
    ],
    "INDUSTRY": [
        "industrial_policy_and_competitiveness",
        "manufacturing_upgrading_and_innovation",
        "resource_efficiency_and_circular_economy",
        "industrial_energy_efficiency",
        "industrial_pollution_control",
        "industrial_parks_and_zones",
        "sme_industrial_development",
        "workforce_skills_and_industry_4_0",
        "industrial_safety_and_labor_standards",
        "industrial_value_chains_and_export",
    ],
    "PRIVATE_FIRMS_AND_SME_DEVELOPMENT": [
        "business_environment_and_regulation",
        "entrepreneurship_and_startups",
        "sme_access_to_finance",
        "business_development_services",
        "value_chain_and_cluster_development",
        "corporate_governance_and_transparency",
        "innovation_and_productivity_support",
        "women_led_and_inclusive_enterprises",
        "digitalization_of_smes",
    ],
    "REFORM_MODERNIZATION_OF_THE_STATE": [
        "public_financial_management",
        "tax_policy_and_administration",
        "civil_service_reform_and_hr",
        "decentralization_and_local_governance",
        "e_government_and_digital_transformation",
        "transparency_anticorruption_and_accountability",
        "regulatory_quality_and_oversight",
        "justice_sector_and_rule_of_law",
        "public_investment_management",
        "state_owned_enterprise_reform",
    ],
    "REGIONAL_INTEGRATION": [
        "cross_border_trade_facilitation",
        "regional_infrastructure_connectivity",
        "regional_energy_markets",
        "regional_financial_integration",
        "migration_and_labor_mobility",
        "regional_environmental_cooperation",
        "regional_institutional_frameworks",
        "regional_security_and_resilience",
    ],
    "SCIENCE_AND_TECHNOLOGY": [
        "research_and_development_systems",
        "innovation_policy_and_startup_ecosystems",
        "digital_infrastructure_and_broadband",
        "data_governance_and_cybersecurity",
        "technology_transfer_and_commercialization",
        "stem_education_and_skills",
        "govtech_and_public_sector_innovation",
        "climate_and_green_technology",
    ],
    "SOCIAL_INVESTMENT": [
        "social_protection_and_safety_nets",
        "poverty_targeting_and_inclusion",
        "labor_market_programs_and_skills",
        "gender_equality_and_womens_empowerment",
        "youth_employment_and_inclusion",
        "indigenous_peoples_and_vulnerable_groups",
        "housing_subsidies_and_social_programs",
        "community_driven_development",
        "social_cohesion_and_conflict_prevention",
    ],
    "SUSTAINABLE_TOURISM": [
        "sustainable_tourism_planning_and_zoning",
        "eco_tourism_and_nature_based_tourism",
        "cultural_heritage_preservation",
        "tourism_value_chains_and_smes",
        "tourism_resilience_and_disaster_risk",
        "tourism_environmental_management",
        "community_based_tourism_and_inclusion",
        "tourism_infrastructure_and_services",
        "tourism_governance_and_destination_management",
    ],
    "TRADE": [
        "trade_policy_and_tariff_reform",
        "trade_facilitation_and_customs",
        "export_promotion_and_diversification",
        "trade_in_services_and_digital_trade",
        "regional_trade_agreements",
        "standards_and_quality_infrastructure",
        "trade_finance_and_logistics",
        "inclusive_trade_msmes_and_gender",
        "trade_adjustment_and_competitiveness",
    ],
    "URBAN_DEVELOPMENT_AND_HOUSING": [
        "urban_planning_and_land_use",
        "affordable_housing_and_slum_upgrading",
        "urban_transport_and_mobility",
        "municipal_services_water_waste_energy",
        "urban_resilience_and_disaster_risk",
        "smart_cities_and_digital_urban_services",
        "urban_governance_and_municipal_finance",
        "public_spaces_and_urban_environment",
        "social_inclusion_and_informal_settlements",
    ],
    "WATER_AND_SANITATION": [
        "water_supply_and_distribution",
        "wastewater_treatment_and_reuse",
        "sanitation_and_hygiene_behaviors",
        "integrated_water_resources_management",
        "irrigation_and_multiuse_water_systems",
        "water_quality_and_pollution_control",
        "flood_risk_management_and_drainage",
        "water_utilities_governance_and_tariffs",
        "rural_water_and_sanitation_access",
        "climate_resilient_water_infrastructure",
    ],
}

# Human-readable display names for sectors
SECTOR_DISPLAY = {
    "TRANSPORT": "Transport",
    "AGRICULTURE_AND_RURAL_DEVELOPMENT": "Agriculture & Rural Development",
    "EDUCATION": "Education",
    "ENERGY": "Energy",
    "ENVIRONMENT_AND_NATURAL_DISASTERS": "Environment & Natural Disasters",
    "FINANCIAL_MARKETS": "Financial Markets",
    "HEALTH": "Health",
    "INDUSTRY": "Industry",
    "PRIVATE_FIRMS_AND_SME_DEVELOPMENT": "Private Firms & SME Development",
    "REFORM_MODERNIZATION_OF_THE_STATE": "Reform & Modernization of the State",
    "REGIONAL_INTEGRATION": "Regional Integration",
    "SCIENCE_AND_TECHNOLOGY": "Science & Technology",
    "SOCIAL_INVESTMENT": "Social Investment",
    "SUSTAINABLE_TOURISM": "Sustainable Tourism",
    "TRADE": "Trade",
    "URBAN_DEVELOPMENT_AND_HOUSING": "Urban Development & Housing",
    "WATER_AND_SANITATION": "Water & Sanitation",
}

# Rich descriptions used as embedding anchor for sector detection
# Each string blends sector name with its most distinctive concepts
SECTOR_DESCRIPTIONS = {
    "TRANSPORT": (
        "transport infrastructure roads highways bridges railways buses metro "
        "vehicles electrification fuels emissions mobility freight logistics "
        "road safety traffic congestion urban transit"
    ),
    "AGRICULTURE_AND_RURAL_DEVELOPMENT": (
        "agriculture farming crops livestock irrigation rural land soil "
        "food security value chains markets smallholders extension services "
        "agroforestry biodiversity pesticides rural development"
    ),
    "EDUCATION": (
        "education schools teachers students curriculum learning outcomes "
        "vocational training universities early childhood literacy numeracy "
        "school infrastructure digital learning inclusive education"
    ),
    "ENERGY": (
        "energy electricity power generation renewable solar wind hydro "
        "grid fossil fuels coal gas efficiency demand clean cooking "
        "electrification energy poverty tariffs subsidies utilities"
    ),
    "ENVIRONMENT_AND_NATURAL_DISASTERS": (
        "environment climate change mitigation adaptation biodiversity "
        "ecosystem protected areas disaster risk natural hazards floods "
        "pollution waste environmental assessment conservation"
    ),
    "FINANCIAL_MARKETS": (
        "finance banking capital markets credit regulation supervision "
        "bonds insurance microfinance SME lending fintech digital payments "
        "anti-money laundering consumer protection financial inclusion"
    ),
    "HEALTH": (
        "health hospitals clinics primary care doctors nurses medicines "
        "disease vaccines maternal child reproductive mental health "
        "sanitation water emergency preparedness health financing"
    ),
    "INDUSTRY": (
        "industry manufacturing production factories industrial parks "
        "competitiveness resource efficiency pollution circular economy "
        "labor standards safety value chains export SME industrial"
    ),
    "PRIVATE_FIRMS_AND_SME_DEVELOPMENT": (
        "private sector firms SMEs small medium enterprises business "
        "entrepreneurship startups investment regulation access to finance "
        "corporate governance innovation women-led enterprises"
    ),
    "REFORM_MODERNIZATION_OF_THE_STATE": (
        "government public administration reform civil service state "
        "fiscal transparency anti-corruption decentralization municipal "
        "e-government digital public financial management tax justice"
    ),
    "REGIONAL_INTEGRATION": (
        "regional integration cross-border trade infrastructure connectivity "
        "energy markets migration labor mobility cooperation agreements "
        "customs harmonization security resilience"
    ),
    "SCIENCE_AND_TECHNOLOGY": (
        "technology innovation research development digital broadband "
        "cybersecurity data startup ecosystem technology transfer STEM "
        "govtech public sector innovation green technology"
    ),
    "SOCIAL_INVESTMENT": (
        "social protection poverty safety nets labor employment gender "
        "equality women youth indigenous vulnerable communities housing "
        "social programs inclusion conflict cohesion"
    ),
    "SUSTAINABLE_TOURISM": (
        "tourism ecotourism sustainable travel heritage cultural nature "
        "destination management hotels hospitality environmental "
        "community-based tourism resilience value chains"
    ),
    "TRADE": (
        "trade export import tariffs customs facilitation logistics "
        "trade agreements services digital trade standards quality "
        "competitiveness MSMEs inclusive trade finance"
    ),
    "URBAN_DEVELOPMENT_AND_HOUSING": (
        "urban city housing slums planning land use municipal services "
        "water waste energy transport mobility smart cities resilience "
        "governance informal settlements public spaces"
    ),
    "WATER_AND_SANITATION": (
        "water supply sanitation sewage wastewater treatment irrigation "
        "water quality pollution flood drainage utilities tariffs "
        "rural water access climate resilient infrastructure WASH"
    ),
}


def label_to_phrase(label_key: str) -> str:
    """Convert 'climate_smart_agriculture' → 'climate smart agriculture'."""
    return label_key.replace("_", " ")


def label_to_display(label_key: str) -> str:
    """Convert 'climate_smart_agriculture' → 'Climate Smart Agriculture'."""
    return label_key.replace("_", " ").title()
