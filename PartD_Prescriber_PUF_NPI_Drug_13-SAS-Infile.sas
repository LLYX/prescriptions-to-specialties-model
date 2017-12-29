
DATA WORK.PartD_Prescriber_PUF_NPI_Drug_13;
	LENGTH
		npi								$ 10
		nppes_provider_last_org_name	$ 70
		nppes_provider_first_name		$ 20
		nppes_provider_city				$ 40
		nppes_provider_state			$  2
		specialty_description			$ 75
		description_flag				$  1
		drug_name		 		 		$ 30
		generic_name					$ 30
		bene_count						   8
		total_claim_count				   8
		total_30_day_fill_count			   8
		total_day_supply				   8
		total_drug_cost					   8
		bene_count_ge65					   8
		bene_count_ge65_suppress_flag	$  1
		total_claim_count_ge65			   8
		ge65_suppress_flag				$  1
		total_30_day_fill_count_ge65	   8
		total_day_supply_ge65			   8
		total_drug_cost_ge65			   8
	;
	
	INFILE 'C:\My documents\PartD_Prescriber_PUF_NPI_Drug_13.txt'
		dlm='09'x
		pad missover
		firstobs = 2
		dsd;

	INPUT
		npi							
		nppes_provider_last_org_name
		nppes_provider_first_name
		nppes_provider_city
		nppes_provider_state
		specialty_description
		description_flag
		drug_name
		generic_name
		bene_count
		total_claim_count
		total_30_day_fill_count
		total_day_supply
		total_drug_cost
		bene_count_ge65
		bene_count_ge65_suppress_flag
		total_claim_count_ge65
		ge65_suppress_flag
		total_30_day_fill_count_ge65
		total_day_supply_ge65
		total_drug_cost_ge65
	;	

	LABEL
		npi								= "National Provider Identifier"
		nppes_provider_last_org_name	= "Last Name/Organization Name of the Provider"
		nppes_provider_first_name		= "First Name of the Provider"
		nppes_provider_city				= "City of the Provider"
		nppes_provider_state			= "State Code of the Provider"
		specialty_description			= "Provider Specialty Type"
		description_flag				= "Source of Provider Specialty"
		drug_name		 		 		= "Brand Name"
		generic_name					= "USAN Generic Name - Short Version"
		bene_count						= "Number of Medicare Beneficiaries"
		total_claim_count 				= "Number of Medicare Part D Claims, Including Refills"
		total_30_day_fill_count			= "Number of Standardized 30-Day Fills, Including Refills"
		total_day_supply 				= "Number of Day's Supply for All Claims"
		total_drug_cost 				= "Aggregate Cost Paid for All Claims"
		bene_count_ge65					= "Number of Medicare Beneficiaries Age 65+"
		bene_count_ge65_suppress_flag	= "Reason for Suppression of Bene_Count_Ge65"
		total_claim_count_ge65			= "Number of Claims, Including Refills, for Beneficiaries Age 65+"
		ge65_suppress_flag				= "Reason for Suppression of Total_Claim_Count_Ge65, Total_30_Day_Fill_Count_Ge65, Total_Day_Supply_Ge65 and Total_Drug_Cost_Ge65"
		total_30_day_fill_count_ge65	= "Number of Standardized 30-Day Fills, Including Refills, for Beneficiaries Age 65+"
		total_day_supply_ge65			= "Number of Day's Supply for All Claims for Beneficiaries Age 65+"
		total_drug_cost_ge65			= "Aggregate Cost Paid for All Claims for Beneficiaries Age 65+"
;

RUN;
