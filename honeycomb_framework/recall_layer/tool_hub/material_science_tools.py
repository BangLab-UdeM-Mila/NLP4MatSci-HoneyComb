import math
from scipy.constants import g
from langchain.tools import tool

@tool
def calculate_degrees_of_freedom(input_params: str) -> int:
    """
    Calculate the degrees of freedom in a system using the number of components and phases.

    Parameters:
    - input_params (str): A single string containing all necessary parameters separated by "|" in the following order:
        components|phases
        - components (int): Number of components in the system.
        - phases (int): Number of phases in the system.

    Returns:
    - int: Degrees of freedom in the system.
    """
    components, phases = map(eval, input_params.split('|'))
    return components - phases + 2

@tool
def calculate_max_phases(input_params: str) -> int:
    """
    Calculate the maximum number of phases that can coexist in a system at constant pressure using Gibbs phase rule.

    Parameters:
    - input_params (str): A single string containing all necessary parameters separated by "|" in the following order:
        components|degrees_of_freedom
        - components (int): Number of components in the system.
        - degrees_of_freedom (int): Degrees of freedom at equilibrium.

    Returns:
    - int: Maximum number of phases that can coexist under the given conditions.
    """
    components, degrees_of_freedom = map(eval, input_params.split('|'))
    return components - degrees_of_freedom + 2

@tool
def calculate_weight_fractions(input_params: str) -> str:
    """
    Calculate the weight fractions of α and β phases based on their ratio.

    Parameters:
    - input_params (str): A single string containing one parameter separated by "|" in the following order:
        ratio_alpha_beta
        - ratio_alpha_beta (float): The ratio of weight percentages of α to β.

    Returns:
    - str: A string containing the weight fractions of α and β phases separated by "|".
    """
    ratio_alpha_beta = eval(input_params.split('|')[0])
    weight_fraction_alpha = ratio_alpha_beta
    weight_fraction_beta = 1
    return f'{weight_fraction_alpha}|{weight_fraction_beta}'

@tool
def calculate_total_weight_percent_a(input_params: str) -> float:
    """
    Calculate the total weight percentage of A in the alloy.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        weight_fraction_alpha|weight_fraction_beta|wt_percent_A_alpha|wt_percent_A_beta
        - weight_fraction_alpha (float): Weight fraction of α phase.
        - weight_fraction_beta (float): Weight fraction of β phase.
        - wt_percent_A_alpha (float): The wt.% of A in α phase.
        - wt_percent_A_beta (float): The wt.% of A in β phase.

    Returns:
    - float: The total weight percentage of A in the alloy.
    """
    params = list(map(eval, input_params.split('|')))
    weight_fraction_alpha, weight_fraction_beta, wt_percent_A_alpha, wt_percent_A_beta = params
    total_weight_fraction = weight_fraction_alpha + weight_fraction_beta
    wt_percent_A_total = (weight_fraction_alpha * wt_percent_A_alpha + weight_fraction_beta * wt_percent_A_beta) / total_weight_fraction
    return wt_percent_A_total

@tool
def calculate_weight_percent_b_in_alloy(input_params: str) -> int:
    """
    Calculate the weight percentage of B in the alloy.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        wt_percent_A_total
        - wt_percent_A_total (float): The total weight percentage of A in the alloy.

    Returns:
    - int: The weight percentage of B in the alloy, rounded to the nearest integer.
    """
    wt_percent_A_total = eval(input_params.split('|')[0])
    wt_percent_B_total = 100 - wt_percent_A_total
    return round(wt_percent_B_total)

@tool
def calculate_mass_fractions_alpha(input_params: str) -> str:
    """
    Calculate the mass fractions of A and B in phase α.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        weight_percent_A_alpha|weight_percent_B_alpha
        - weight_percent_A_alpha (float): The weight percentage of A in phase α.
        - weight_percent_B_alpha (float): The weight percentage of B in phase α.

    Returns:
    - str: A string containing the mass fractions of A and B in phase α separated by "|".
    """
    weight_percent_A_alpha, weight_percent_B_alpha = map(float, input_params.split('|'))
    total_weight_alpha = weight_percent_A_alpha + weight_percent_B_alpha
    mass_fraction_A_alpha = weight_percent_A_alpha / total_weight_alpha
    mass_fraction_B_alpha = weight_percent_B_alpha / total_weight_alpha
    return f'{mass_fraction_A_alpha}|{mass_fraction_B_alpha}'

@tool
def calculate_moles_alpha(input_params: str) -> str:
    """
    Calculate the moles of A and B in phase α.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        mass_fraction_A_alpha|mass_fraction_B_alpha|atomic_mass_A|atomic_mass_B
        - mass_fraction_A_alpha (float): Mass fraction of A in phase α.
        - mass_fraction_B_alpha (float): Mass fraction of B in phase α.
        - atomic_mass_A (float): Atomic mass of A.
        - atomic_mass_B (float): Atomic mass of B.

    Returns:
    - str: A string containing the moles of A and B in phase α separated by "|".
    """
    mass_fraction_A_alpha, mass_fraction_B_alpha, atomic_mass_A, atomic_mass_B = map(float, input_params.split('|'))
    moles_A_alpha = mass_fraction_A_alpha / atomic_mass_A
    moles_B_alpha = mass_fraction_B_alpha / atomic_mass_B
    return f'{moles_A_alpha}|{moles_B_alpha}'

@tool
def calculate_mole_fractions_beta(input_params: str) -> str:
    """
    Calculate the mole fractions of A and B in phase β.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        mole_ratio_A_to_B_beta
        - mole_ratio_A_to_B_beta (float): The mole fraction ratio of A to B in phase β.

    Returns:
    - str: A string containing the mole fractions of A and B in phase β separated by "|".
    """
    mole_ratio_A_to_B_beta = float(input_params.split('|')[0])
    mole_fraction_A_beta = mole_ratio_A_to_B_beta / (1 + mole_ratio_A_to_B_beta)
    mole_fraction_B_beta = 1 - mole_fraction_A_beta
    return f'{mole_fraction_A_beta}|{mole_fraction_B_beta}'

@tool
def calculate_moles_beta(input_params: str) -> str:
    """
    Calculate the moles of A and B in phase β assuming equal mass for α and β phases.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        total_weight_alpha|weight_ratio_alpha_to_beta|mole_fraction_A_beta|mole_fraction_B_beta|atomic_mass_A|atomic_mass_B
        - total_weight_alpha (float): Total weight of phase α.
        - weight_ratio_alpha_to_beta (float): Weight ratio of phase α to phase β.
        - mole_fraction_A_beta (float): Mole fraction of A in phase β.
        - mole_fraction_B_beta (float): Mole fraction of B in phase β.
        - atomic_mass_A (float): Atomic mass of A.
        - atomic_mass_B (float): Atomic mass of B.

    Returns:
    - str: A string containing the moles of A and B in phase β separated by "|".
    """
    total_weight_alpha, weight_ratio_alpha_to_beta, mole_fraction_A_beta, mole_fraction_B_beta, atomic_mass_A, atomic_mass_B = map(float, input_params.split('|'))
    total_weight_beta = total_weight_alpha * weight_ratio_alpha_to_beta
    moles_A_beta = total_weight_beta * mole_fraction_A_beta / atomic_mass_A
    moles_B_beta = total_weight_beta * mole_fraction_B_beta / atomic_mass_B
    return f'{moles_A_beta}|{moles_B_beta}'

@tool
def calculate_pearlite_fraction(args: str) -> float:
    """
    Calculate the mass fraction of pearlite just below the eutectoid temperature in an iron-carbon alloy.

    Parameters:
    - args (str): A string containing the required parameters separated by "|" in the following order:
        eutectoid_composition|ferrite_carbon_content|alloy_carbon_content
        - eutectoid_composition (float): Eutectoid composition in wt.% carbon.
        - ferrite_carbon_content (float): Carbon content in ferrite in wt.%.
        - alloy_carbon_content (float): Carbon content in the alloy in wt.%.

    Returns:
    - float: Mass fraction of pearlite, rounded to two decimal places.
    """
    eutectoid_composition, ferrite_carbon_content, alloy_carbon_content = map(float, args.split('|'))
    pearlite_fraction = (alloy_carbon_content - ferrite_carbon_content) / (eutectoid_composition - ferrite_carbon_content)
    return round(pearlite_fraction, 2)

@tool
def calculate_lever_rule_numerator(args: str) -> float:
    """
    Calculate the numerator for the lever rule used in phase fraction calculations.

    Parameters:
    - args (str): A string containing the required parameters separated by "|" in the following order:
        alloy_carbon_content|ferrite_carbon_content
        - alloy_carbon_content (float): Carbon content in the alloy in wt.%.
        - ferrite_carbon_content (float): Carbon content in ferrite in wt.%.
    
    Returns:
    - float: Numerator value for the lever rule.
    """
    alloy_carbon_content, ferrite_carbon_content = map(float, args.split('|'))
    return alloy_carbon_content - ferrite_carbon_content

@tool
def calculate_lever_rule_denominator(args: str) -> float:
    """
    Calculate the denominator for the lever rule used in phase fraction calculations.

    Parameters:
    - args (str): A string containing the required parameters separated by "|" in the following order:
        eutectoid_composition|ferrite_carbon_content
        - eutectoid_composition (float): Eutectoid composition in wt.% carbon.
        - ferrite_carbon_content (float): Carbon content in ferrite in wt.%.
    
    Returns:
    - float: Denominator value for the lever rule.
    """
    eutectoid_composition, ferrite_carbon_content = map(float, args.split('|'))
    return eutectoid_composition - ferrite_carbon_content

@tool
def calculate_rate_constant(args: str) -> float:
    """
    Calculate the rate constant k using the initial conditions of the Avrami equation.

    Parameters:
    - args (str): A single string containing the required parameters separated by "|" in the following order:
        n|t1|X_t1
        - n (float): Avrami exponent.
        - t1 (float): Initial time in seconds.
        - X_t1 (float): Recrystallization percentage at initial time (as a decimal, e.g., 0.20 for 20%).

    Returns:
    - float: The rate constant k.
    """
    import math
    n, t1, X_t1 = map(float, args.split('|'))
    k = -math.log(1 - X_t1) / t1 ** n
    return k

@tool
def calculate_recrystallization_percentage(args: str) -> float:
    """
    Calculate the recrystallization percentage at final time using the Avrami equation and the rate constant k.

    Parameters:
    - args (str): A single string containing the required parameters separated by "|" in the following order:
        k|n|t2
        - k (float): The rate constant.
        - n (float): Avrami exponent.
        - t2 (float): Final time in seconds.

    Returns:
    - float: Recrystallization percentage at final time, rounded to two decimal places.
    """
    import math
    k, n, t2 = map(float, args.split('|'))
    X_t2 = 1 - math.exp(-k * t2 ** n)
    return round(X_t2 * 100, 2)

@tool
def convert_pressure_atm_to_pa(input_params: str) -> float:
    """
    Convert pressure from atm to Pa.

    Parameters:
    - input_params (str): A single string containing the pressure in atm separated by "|":
        P1
        - P1 (float): Initial pressure in atm.

    Returns:
    - float: Pressure in Pa.
    """
    P1 = float(input_params.split('|')[0])
    P1_pa = P1 * 101325.0
    return P1_pa

@tool
def convert_volume_cm3_to_m3(input_params: str) -> str:
    """
    Convert volume from cm³/mol to m³/mol.

    Parameters:
    - input_params (str): A single string containing the volumes in cm³/mol separated by "|":
        V_alpha|V_beta
        - V_alpha (float): Molar volume of α phase in cm³/mol.
        - V_beta (float): Molar volume of β phase in cm³/mol.

    Returns:
    - str: Volumes in m³/mol separated by "|".
    """
    V_alpha, V_beta = map(float, input_params.split('|'))
    V_alpha_m3 = V_alpha * 1e-06
    V_beta_m3 = V_beta * 1e-06
    return f'{V_alpha_m3}|{V_beta_m3}'

@tool
def calculate_change_in_volume(input_params: str) -> float:
    """
    Calculate the change in volume ΔV.

    Parameters:
    - input_params (str): A single string containing the volumes in m³/mol separated by "|":
        V_alpha_m3|V_beta_m3
        - V_alpha_m3 (float): Molar volume of α phase in m³/mol.
        - V_beta_m3 (float): Molar volume of β phase in m³/mol.

    Returns:
    - float: Change in volume ΔV in m³/mol.
    """
    V_alpha_m3, V_beta_m3 = map(float, input_params.split('|'))
    delta_V = V_beta_m3 - V_alpha_m3
    return delta_V

@tool
def calculate_change_in_entropy(input_params: str) -> float:
    """
    Calculate the change in entropy ΔS.

    Parameters:
    - input_params (str): A single string containing the enthalpy and initial temperature separated by "|":
        delta_H|T1
        - delta_H (float): Enthalpy of transformation in J/mol.
        - T1 (float): Initial temperature in K.

    Returns:
    - float: Change in entropy ΔS in J/(mol·K).
    """
    delta_H, T1 = map(float, input_params.split('|'))
    delta_S = delta_H / T1
    return delta_S

@tool
def calculate_change_in_pressure(input_params: str) -> float:
    """
    Calculate the pressure change ΔP.

    Parameters:
    - input_params (str): A single string containing the enthalpy, volumes, and temperatures separated by "|":
        delta_H|delta_V|T1|T2
        - delta_H (float): Enthalpy of transformation in J/mol.
        - delta_V (float): Change in volume in m³/mol.
        - T1 (float): Initial temperature in K.
        - T2 (float): Final temperature in K.

    Returns:
    - float: Pressure change ΔP in Pa.
    """
    delta_H, delta_V, T1, T2 = map(float, input_params.split('|'))
    delta_P = delta_H / delta_V * (1 / T2 - 1 / T1)
    return delta_P

@tool
def convert_pressure_pa_to_atm(input_params: str) -> float:
    """
    Convert pressure from Pa to atm.

    Parameters:
    - input_params (str): A single string containing the pressure in Pa separated by "|":
        P_pa
        - P_pa (float): Pressure in Pa.

    Returns:
    - float: Pressure in atm.
    """
    P_pa = float(input_params.split('|')[0])
    P_atm = P_pa / 101325.0
    return P_atm

@tool
def calculate_bcc_unit_cell_volume(input_params: str) -> float:
    """
    Calculate the volume of the BCC unit cell for Titanium (Ti).

    Parameters:
    - input_params (str): A single string containing the lattice parameter of bcc Ti in nm separated by "|":
        lattice_param_bcc
        - lattice_param_bcc (float): Lattice parameter of bcc Ti in nm.

    Returns:
    - float: Volume of the BCC unit cell in cm³.
    """
    lattice_param_bcc = float(input_params.split('|')[0])
    volume_bcc = (lattice_param_bcc * 1e-07) ** 3
    return volume_bcc

@tool
def calculate_hcp_unit_cell_volume(input_params: str) -> float:
    """
    Calculate the volume of the HCP unit cell for Titanium (Ti).

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        atomic_weight|density_hcp|avogadro_number
        - atomic_weight (float): Atomic weight of Ti.
        - density_hcp (float): Density of hcp Ti in g/cm³.
        - avogadro_number (float): Avogadro's number.

    Returns:
    - float: Volume of the HCP unit cell in cm³.
    """
    atomic_weight, density_hcp, avogadro_number = map(float, input_params.split('|'))
    Z_hcp = 2
    volume_hcp = Z_hcp * atomic_weight / (density_hcp * avogadro_number)
    return volume_hcp

@tool
def calculate_volume_change_percentage(input_params: str) -> float:
    """
    Calculate the volume change percentage when Titanium (Ti) transforms from HCP to BCC.

    Parameters:
    - input_params (str): A single string containing the volumes of BCC and HCP unit cells separated by "|":
        volume_bcc|volume_hcp
        - volume_bcc (float): Volume of the BCC unit cell in cm³.
        - volume_hcp (float): Volume of the HCP unit cell in cm³.

    Returns:
    - float: Volume change percentage, rounded to one decimal place.
    """
    volume_bcc, volume_hcp = map(float, input_params.split('|'))
    volume_change_percentage = (volume_bcc - volume_hcp) / volume_hcp * 100
    return round(volume_change_percentage, 1)

@tool
def calculate_bcc_unit_cell_volume(input_params: str) -> float:
    """
    Calculate the volume of the BCC unit cell for iron (Fe).

    Parameters:
    - input_params (str): A single string containing the lattice parameter of BCC phase in nm separated by "|":
        lattice_param_bcc
        - lattice_param_bcc (float): Lattice parameter of BCC phase in nm.

    Returns:
    - float: Volume of the BCC unit cell in cm³.
    """
    lattice_param_bcc = float(input_params.split('|')[0])
    volume_bcc = (lattice_param_bcc * 1e-07) ** 3 * 2
    return volume_bcc

@tool
def calculate_fcc_unit_cell_volume(input_params: str) -> float:
    """
    Calculate the volume of the FCC unit cell for iron (Fe).

    Parameters:
    - input_params (str): A single string containing the lattice parameter of FCC phase in nm separated by "|":
        lattice_param_fcc
        - lattice_param_fcc (float): Lattice parameter of FCC phase in nm.

    Returns:
    - float: Volume of the FCC unit cell in cm³.
    """
    lattice_param_fcc = float(input_params.split('|')[0])
    volume_fcc = (lattice_param_fcc * 1e-07) ** 3 * 4
    return volume_fcc

@tool
def calculate_volume_change_percentage(input_params: str) -> float:
    """
    Calculate the volume change percentage when iron (Fe) transforms from BCC to FCC.

    Parameters:
    - input_params (str): A single string containing the volumes of BCC and FCC unit cells separated by "|":
        volume_bcc|volume_fcc
        - volume_bcc (float): Volume of the BCC unit cell in cm³.
        - volume_fcc (float): Volume of the FCC unit cell in cm³.

    Returns:
    - float: Volume change percentage, rounded to one decimal place.
    """
    volume_bcc, volume_fcc = map(float, input_params.split('|'))
    volume_change_percentage = (volume_fcc - volume_bcc) / volume_bcc * 100
    return round(volume_change_percentage, 1)

@tool
def calculate_temperature_ratio(input_params: str) -> float:
    """
    Calculate the temperature ratio (T/Tm).

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        temperature|melting_point
        - temperature (float): Temperature at which transformation occurs in K.
        - melting_point (float): Melting point of the substance in K.

    Returns:
    - float: Temperature ratio (T/Tm).
    """
    temperature, melting_point = map(float, input_params.split('|'))
    temperature_ratio = temperature / melting_point
    return temperature_ratio

@tool
def calculate_transformation_driving_force(input_params: str) -> float:
    """
    Calculate the driving force for liquid to solid transformation.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        enthalpy_of_melting|temperature_ratio
        - enthalpy_of_melting (float): Enthalpy of melting in J/mol.
        - temperature_ratio (float): Temperature ratio (T/Tm).

    Returns:
    - float: Driving force for transformation in J/mol, rounded to one decimal place.
    """
    enthalpy_of_melting, temperature_ratio = map(float, input_params.split('|'))
    delta_G = enthalpy_of_melting * (1 - temperature_ratio) * 1000
    return round(delta_G, 1)

@tool
def calculate_driving_force(input_params: str) -> float:
    """
    Calculate the driving force for liquid to solid transformation at a given temperature.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        melting_point|enthalpy_of_melting|temperature
        - melting_point (float): Melting point of the substance in K.
        - enthalpy_of_melting (float): Enthalpy of melting in J/mol.
        - temperature (float): Temperature at which transformation occurs in K.

    Returns:
    - float: Driving force for transformation in J/mol, rounded to one decimal place.
    """
    melting_point, enthalpy_of_melting, temperature = input_params.split('|')
    temperature_ratio = calculate_temperature_ratio(f'{temperature}|{melting_point}')
    driving_force = calculate_transformation_driving_force(f'{enthalpy_of_melting}|{temperature_ratio}')
    return driving_force

@tool
def calculate_temperature_ratio(input_params: str) -> float:
    """
    Calculate the temperature ratio (T/Tm).

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        temperature|melting_point
        - temperature (float): Temperature at which transformation occurs in K.
        - melting_point (float): Melting point of the substance in K.

    Returns:
    - float: Temperature ratio (T/Tm).
    """
    temperature, melting_point = map(float, input_params.split('|'))
    temperature_ratio = temperature / melting_point
    return temperature_ratio

@tool
def calculate_free_energy(input_params: str) -> float:
    """
    Calculate the free energy change for liquid to solid transformation.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        enthalpy_of_melting|temperature_ratio
        - enthalpy_of_melting (float): Enthalpy of melting in J/mol.
        - temperature_ratio (float): Temperature ratio (T/Tm).

    Returns:
    - float: Free energy change in J/mol.
    """
    enthalpy_of_melting, temperature_ratio = map(float, input_params.split('|'))
    delta_G = enthalpy_of_melting * (1 - temperature_ratio)
    return delta_G

@tool
def calculate_carbon_difference(input_params: str) -> float:
    """
    Calculate the difference in carbon content between the eutectoid composition and the carbon content of the steel sample.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        eutectoid_composition|carbon_content
        - eutectoid_composition (float): Eutectoid composition in wt.% C.
        - carbon_content (float): Carbon content of the steel sample in wt.% C.

    Returns:
    - float: Difference in carbon content.
    """
    eutectoid_composition, carbon_content = map(float, input_params.split('|'))
    carbon_difference = eutectoid_composition - carbon_content
    return carbon_difference

@tool
def calculate_fraction_ferrite(input_params: str) -> float:
    """
    Calculate the fraction of proeutectoid ferrite in a steel sample.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        carbon_difference|lever_rule_denominator
        - carbon_difference (float): Difference in carbon content between the eutectoid composition and the carbon content of the steel sample.
        - lever_rule_denominator (float): Difference between the eutectoid composition and the maximum solubility of carbon in α-Fe.

    Returns:
    - float: Fraction of proeutectoid ferrite in the microstructure.
    """
    carbon_difference, lever_rule_denominator = map(float, input_params.split('|'))
    fraction_ferrite = carbon_difference / lever_rule_denominator
    return fraction_ferrite

@tool
def convert_temperature_to_kelvin(input_params: str) -> float:
    """
    Convert temperature from °C to K.

    Parameters:
    - input_params (str): A single string containing the temperature in °C separated by "|":
        temperature
        - temperature (float): Temperature in °C.

    Returns:
    - float: Temperature in K.
    """
    temperature = float(input_params.split('|')[0])
    temperature_kelvin = temperature + 273.15
    return temperature_kelvin

@tool
def calculate_standard_cell_potential(input_params: str) -> float:
    """
    Calculate the standard cell potential for a galvanic cell.

    Parameters:
    - input_params (str): A single string containing the standard electrode potentials separated by "|" in the following order:
        standard_potential_cd|standard_potential_ni
        - standard_potential_cd (float): Standard electrode reduction potential of Cd in V.
        - standard_potential_ni (float): Standard electrode reduction potential of Ni in V.

    Returns:
    - float: Standard cell potential in V.
    """
    standard_potential_cd, standard_potential_ni = map(float, input_params.split('|'))
    standard_cell_potential = standard_potential_cd - standard_potential_ni
    return standard_cell_potential

@tool
def calculate_ln_q(input_params: str) -> float:
    """
    Calculate the natural logarithm of the concentration ratio.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        standard_cell_potential|temperature_kelvin|gas_constant|faraday_constant
        - standard_cell_potential (float): Standard cell potential in V.
        - temperature_kelvin (float): Temperature in K.
        - gas_constant (float): Universal gas constant in J/(mol·K).
        - faraday_constant (float): Faraday's constant in C/mol.

    Returns:
    - float: Natural logarithm of the concentration ratio.
    """
    standard_cell_potential, temperature_kelvin, gas_constant, faraday_constant = map(float, input_params.split('|'))
    ln_q = standard_cell_potential * 2 * faraday_constant / (gas_constant * temperature_kelvin)
    return ln_q

@tool
def calculate_conductivity(input_params: str) -> float:
    """
    Calculate the electrical conductivity of copper.

    Parameters:
    - input_params (str): A single string containing the resistivity separated by "|":
        resistivity
        - resistivity (float): Electrical resistivity of copper in Ω·m.

    Returns:
    - float: Electrical conductivity of copper in S/m.
    """
    resistivity = float(input_params.split('|')[0])
    conductivity = 1 / resistivity
    return conductivity

@tool
def calculate_free_electron_concentration(input_params: str) -> float:
    """
    Calculate the free electron concentration in copper.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        conductivity|electron_charge|electron_mobility
        - conductivity (float): Electrical conductivity of copper in S/m.
        - electron_charge (float): Charge of an electron in C.
        - electron_mobility (float): Electron mobility in m²/(V·s).

    Returns:
    - float: Free electron concentration in copper in electrons/m³.
    """
    conductivity, electron_charge, electron_mobility = map(float, input_params.split('|'))
    free_electron_concentration = conductivity / (electron_charge * electron_mobility)
    return free_electron_concentration

@tool
def calculate_atomic_density(input_params: str) -> float:
    """
    Calculate the atomic density of copper.

    Parameters:
    - input_params (str): A single string containing the lattice parameter separated by "|":
        lattice_parameter
        - lattice_parameter (float): Lattice parameter of copper in meters.

    Returns:
    - float: Atomic density of copper in atoms/m³.
    """
    lattice_parameter = float(input_params.split('|')[0])
    V_cell = lattice_parameter ** 3
    atomic_density = 4 / V_cell
    return atomic_density

@tool
def calculate_potential_vs_she(input_params: str) -> float:
    """
    Calculate the reaction potential versus the Standard Hydrogen Electrode (SHE).

    Parameters:
    - input_params (str): A single string containing the standard potential of Li and the reaction potential against the Li reference electrode separated by "|":
        e_li|e_reaction_li
        - e_li (float): Standard potential of Li in V.
        - e_reaction_li (float): Reaction potential against Li reference electrode in V.

    Returns:
    - float: Reaction potential versus SHE in V.
    """
    e_li, e_reaction_li = map(float, input_params.split('|'))
    e_reaction_she = e_reaction_li + e_li
    return e_reaction_she

@tool
def calculate_potential_vs_zn(input_params: str) -> float:
    """
    Calculate the reaction potential versus Zn²⁺/Zn reference electrode.

    Parameters:
    - input_params (str): A single string containing the reaction potential versus SHE and the standard potential of Zn separated by "|":
        e_reaction_she|e_zn
        - e_reaction_she (float): Reaction potential versus SHE in V.
        - e_zn (float): Standard potential of Zn in V.

    Returns:
    - float: Reaction potential versus Zn²⁺/Zn in V.
    """
    e_reaction_she, e_zn = map(float, input_params.split('|'))
    e_reaction_zn = e_reaction_she - e_zn
    return e_reaction_zn

@tool
def calculate_total_copper_mass(input_params: str) -> float:
    """
    Calculate the total mass of copper in grams from the copper concentrate.

    Parameters:
    - input_params (str): A single string containing the total mass of concentrate and copper content separated by "|":
        total_mass_concentrate|copper_content
        - total_mass_concentrate (float): Total mass of copper concentrate in metric tons.
        - copper_content (float): Copper content in wt.%.

    Returns:
    - float: Total mass of copper in grams.
    """
    total_mass_concentrate, copper_content = map(float, input_params.split('|'))
    total_mass_concentrate_g = total_mass_concentrate * 1000000.0
    total_copper_mass = total_mass_concentrate_g * (copper_content / 100)
    return total_copper_mass

@tool
def calculate_moles_of_copper(input_params: str) -> float:
    """
    Calculate the moles of copper from the total mass of copper.

    Parameters:
    - input_params (str): A single string containing the total copper mass and atomic weight of copper separated by "|":
        total_copper_mass|atomic_weight_Cu
        - total_copper_mass (float): Total mass of copper in grams.
        - atomic_weight_Cu (float): Atomic weight of copper in g/mol.

    Returns:
    - float: Moles of copper.
    """
    total_copper_mass, atomic_weight_Cu = map(float, input_params.split('|'))
    moles_of_copper = total_copper_mass / atomic_weight_Cu
    return moles_of_copper

@tool
def calculate_total_charge(input_params: str) -> float:
    """
    Calculate the total charge in Coulombs required for processing copper.

    Parameters:
    - input_params (str): A single string containing the moles of copper and Faraday constant separated by "|":
        moles_of_copper|Faraday_constant
        - moles_of_copper (float): Moles of copper.
        - Faraday_constant (float): Faraday constant in C/mol.

    Returns:
    - float: Total charge in Coulombs.
    """
    moles_of_copper, Faraday_constant = map(float, input_params.split('|'))
    total_charge = moles_of_copper * 2 * Faraday_constant
    return total_charge

@tool
def calculate_total_working_hours(input_params: str) -> float:
    """
    Calculate the total working hours for the processing period.

    Parameters:
    - input_params (str): A single string containing the number of working days per month and working hours per day separated by "|":
        working_days_per_month|working_hours_per_day
        - working_days_per_month (int): Number of working days per month.
        - working_hours_per_day (int): Number of working hours per day.

    Returns:
    - float: Total working hours.
    """
    working_days_per_month, working_hours_per_day = map(float, input_params.split('|'))
    total_working_hours = working_days_per_month * working_hours_per_day * 6
    return total_working_hours

@tool
def calculate_conductivity_cm(input_params: str) -> float:
    """
    Calculate the electrical conductivity of n-type silicon doped with phosphorus atoms in cm^-1.

    Parameters:
    - input_params (str): A single string containing all necessary parameters separated by "|" in the following order:
        charge_electron|dopant_concentration|electron_mobility
        - charge_electron (float): Charge of an electron in Coulombs.
        - dopant_concentration (float): Concentration of phosphorus atoms in cm^-3.
        - electron_mobility (float): Drift mobility of electrons in cm^2/V.s.

    Returns:
    - float: Electrical conductivity in Ω^-1.cm^-1.
    """
    charge_electron, dopant_concentration, electron_mobility = map(float, input_params.split('|'))
    conductivity = charge_electron * dopant_concentration * electron_mobility
    return conductivity

@tool
def convert_conductivity_to_m(input_params: str) -> float:
    """
    Convert electrical conductivity from Ω^-1.cm^-1 to Ω^-1.m^-1.

    Parameters:
    - input_params (str): A single string containing the electrical conductivity in cm^-1:
        conductivity_cm
        - conductivity_cm (float): Electrical conductivity in Ω^-1.cm^-1.

    Returns:
    - float: Electrical conductivity in Ω^-1.m^-1.
    """
    conductivity_cm = float(input_params)
    conductivity_m = conductivity_cm * 100.0
    return conductivity_m

@tool
def calculate_reaction_quotient(args: str) -> float:
    """
    Calculate the reaction quotient Q based on the activity of copper in the alloy.

    Parameters:
    - input_params (str): A single string containing the activity of copper:
        a_Cu
        - a_Cu (float): Activity of copper in the alloy (anode).

    Returns:
    - float: Reaction quotient Q.
    """
    a_Cu = float(args)
    return a_Cu

@tool
def calculate_nernst_potential(args: str) -> float:
    """
    Calculate the cell potential using the Nernst equation.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        T|n|F|R|Q
        - T (float): Temperature in K.
        - n (int): Number of electrons transferred.
        - F (float): Faraday constant in C/mol.
        - R (float): Universal gas constant in J/(mol·K).
        - Q (float): Reaction quotient.

    Returns:
    - float: Cell potential in volts.
    """
    T, n, F, R, Q = map(float, args.split('|'))
    E = R * T / (n * F) * math.log(Q)
    return E

@tool
def convert_voltage_to_millivolts(args: str) -> float:
    """
    Convert cell potential from volts to millivolts.

    Parameters:
    - input_params (str): A single string containing the cell potential in volts:
        E
        - E (float): Cell potential in volts.

    Returns:
    - float: Cell potential in millivolts.
    """
    E = float(args)
    E_mV = E * 1000
    return E_mV

@tool
def calculate_reaction_quotient_nickel(args: str) -> float:
    """
    Calculate the reaction quotient Q based on the mole fraction of Ni in the anode.

    Parameters:
    - input_params (str): A single string containing the mole fraction of Ni in the anode:
        mole_fraction_Ni_anode
        - mole_fraction_Ni_anode (float): Mole fraction of Ni in the anode.

    Returns:
    - float: Reaction quotient Q.
    """
    mole_fraction_Ni_anode = float(args)
    return mole_fraction_Ni_anode

@tool
def calculate_nernst_voltage(args: str) -> float:
    """
    Calculate the minimum voltage using the Nernst equation.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        T|n|F|R|Q
        - T (float): Temperature in K.
        - n (int): Number of electrons transferred.
        - F (float): Faraday constant in C/mol.
        - R (float): Universal gas constant in J/(mol·K).
        - Q (float): Reaction quotient.

    Returns:
    - float: Minimum voltage in volts.
    """
    T, n, F, R, Q = map(float, args.split('|'))
    E = R * T / (n * F) * math.log(Q)
    return E

@tool
def convert_voltage_to_millivolts(args: str) -> float:
    """
    Convert voltage from volts to millivolts.

    Parameters:
    - input_params (str): A single string containing the voltage in volts:
        E
        - E (float): Voltage in volts.

    Returns:
    - float: Voltage in millivolts.
    """
    E = float(args)
    E_mV = E * 1000
    return E_mV

@tool
def calculate_charge_passed(args: str) -> float:
    """
    Calculate the charge passed in Coulombs.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        current|time
        - current (float): Current in A.
        - time (float): Time in seconds (assumed to be 1 for this calculation).

    Returns:
    - float: Charge passed in Coulombs.
    """
    current, time = map(float, args.split('|'))
    charge_passed = current * time
    return charge_passed

@tool
def calculate_moles_of_zinc(args: str) -> float:
    """
    Calculate the moles of Zn deposited.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        charge_passed|F|n
        - charge_passed (float): Charge passed in Coulombs.
        - F (float): Faraday constant in C/mol.
        - n (int): Number of electrons transferred.

    Returns:
    - float: Moles of Zn deposited.
    """
    charge_passed, F, n = map(float, args.split('|'))
    moles_Zn = charge_passed / (n * F)
    return moles_Zn

@tool
def calculate_mass_of_zinc(args: str) -> float:
    """
    Calculate the mass of Zn deposited in grams.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        moles_Zn|atomic_weight_Zn
        - moles_Zn (float): Moles of Zn deposited.
        - atomic_weight_Zn (float): Atomic weight of Zn in g/mol.

    Returns:
    - float: Mass of Zn deposited in grams.
    """
    moles_Zn, atomic_weight_Zn = map(float, args.split('|'))
    mass_Zn = moles_Zn * atomic_weight_Zn
    return mass_Zn

@tool
def calculate_actual_mass_of_zinc(args: str) -> float:
    """
    Calculate the actual mass of Zn deposited considering the current efficiency.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        mass_Zn|current_efficiency
        - mass_Zn (float): Mass of Zn deposited in grams.
        - current_efficiency (float): Cathodic current efficiency (as a decimal).

    Returns:
    - float: Actual mass of Zn deposited in grams.
    """
    mass_Zn, current_efficiency = map(float, args.split('|'))
    actual_mass_Zn = mass_Zn * current_efficiency
    return actual_mass_Zn

@tool
def calculate_energy_consumed(args: str) -> float:
    """
    Calculate the energy consumed in kJ.

    Parameters:
    - input_params (str): A single string containing the required parameters separated by "|" in the following order:
        voltage|current|time
        - voltage (float): Voltage in V.
        - current (float): Current in A.
        - time (float): Time in seconds (assumed to be 1 for this calculation).

    Returns:
    - float: Energy consumed in kJ.
    """
    voltage, current, time = map(float, args.split('|'))
    energy_consumed = voltage * current * time
    energy_consumed_kJ = energy_consumed / 1000
    return energy_consumed_kJ

@tool
def convert_oxygen_concentration(args: str) -> float:
    """
    Convert oxygen concentration from mM to mol/cm³.

    Parameters:
    - args (str): A single string containing the oxygen concentration in mM.
    
    Returns:
    - float: Oxygen concentration in mol/cm³.
    """
    oxygen_concentration = float(args)
    oxygen_concentration_mol_cm3 = oxygen_concentration * 0.001 * 0.001
    return oxygen_concentration_mol_cm3

@tool
def calculate_diffusion_limited_current_density(args: str) -> float:
    """
    Calculate the diffusion-limited current density (i_L).

    Parameters:
    - args (str): A single string containing the required parameters separated by "|" in the following order:
        F|diffusion_coefficient_oxygen|oxygen_concentration_mol_cm3|diffusion_layer_thickness
        - F (float): Faraday's constant in C/mol.
        - diffusion_coefficient_oxygen (float): Diffusion coefficient of oxygen in cm^2/s.
        - oxygen_concentration_mol_cm3 (float): Concentration of dissolved oxygen in mol/cm³.
        - diffusion_layer_thickness (float): Diffusion layer thickness in cm.

    Returns:
    - float: Diffusion-limited current density in A/cm^2.
    """
    F, diffusion_coefficient_oxygen, oxygen_concentration_mol_cm3, diffusion_layer_thickness = map(float, args.split('|'))
    i_L = 4 * F * diffusion_coefficient_oxygen * oxygen_concentration_mol_cm3 / diffusion_layer_thickness
    return i_L

@tool
def calculate_anodic_current_density(args: str) -> float:
    """
    Calculate the anodic current density using the Tafel equation.

    Parameters:
    - args (str): A single string containing the required parameters separated by "|" in the following order:
        i_L|tafel_slope|overpotential
        - i_L (float): Diffusion-limited current density in A/cm^2.
        - tafel_slope (float): Anodic Tafel slope in V.
        - overpotential (float): Overpotential in V.

    Returns:
    - float: Anodic current density in A/cm^2.
    """
    i_L, tafel_slope, overpotential = map(float, args.split('|'))
    i_a = i_L * 10 ** (overpotential / tafel_slope)
    return round(i_a, 3)

@tool
def calculate_si_atom_density(args: str) -> float:
    """
    Calculate the number of silicon atoms per m^3.

    Parameters:
    - args (str): A single string containing the required parameters separated by "|" in the following order:
        density_si|avogadro_number|si_molar_mass
        - density_si (float): Density of silicon in g/cm^3.
        - avogadro_number (float): Avogadro's number in mol^-1.
        - si_molar_mass (float): Molar mass of silicon in g/mol.

    Returns:
    - float: Number of silicon atoms per m^3.
    """
    density_si, avogadro_number, si_molar_mass = map(float, args.split('|'))
    density_si_kg_m3 = density_si * 1000
    n_si = density_si_kg_m3 / si_molar_mass
    n_si_atoms = n_si * avogadro_number
    return n_si_atoms

@tool
def calculate_p_atoms_per_m3(args: str) -> float:
    """
    Calculate the number of phosphorus atoms per m^3.

    Parameters:
    - args (str): A single string containing the required parameters separated by "|" in the following order:
        n_si_atoms|residual_p_content
        - n_si_atoms (float): Number of silicon atoms per m^3.
        - residual_p_content (float): Residual phosphorus content in parts per billion by weight.

    Returns:
    - float: Number of phosphorus atoms per m^3.
    """
    n_si_atoms, residual_p_content = map(float, args.split('|'))
    p_atoms_per_m3 = n_si_atoms * (residual_p_content * 1e-09)
    return p_atoms_per_m3

@tool
def calculate_conductivity(args: str) -> float:
    """
    Calculate the electrical conductivity.

    Parameters:
    - args (str): A single string containing the required parameters separated by "|" in the following order:
        charge_electron|mobility_electron|p_atoms_per_m3
        - charge_electron (float): Charge of an electron in A.s (Coulombs).
        - mobility_electron (float): Mobility of electrons in m^2/V.s.
        - p_atoms_per_m3 (float): Number of phosphorus atoms per m^3.

    Returns:
    - float: Electrical conductivity in Ω^-1.m^-1.
    """
    charge_electron, mobility_electron, p_atoms_per_m3 = map(float, args.split('|'))
    conductivity = charge_electron * mobility_electron * p_atoms_per_m3
    return round(conductivity * 1000, 2)

@tool
def calculate_delta_g_standard(args: str) -> float:
    """
    Calculate the Gibbs free energy change (ΔG°) in J·mol⁻¹.

    Parameters:
    - args (str): A single string containing the required parameters separated by "|" in the following order:
        n|F|E_standard
        - n (int): Number of electrons transferred
        - F (float): Faraday constant in C·mol⁻¹
        - E_standard (float): Standard electrode potential (E°) in V

    Returns:
    - float: Gibbs free energy change (ΔG°) in J·mol⁻¹.
    """
    n, F, E_standard = map(float, args.split('|'))
    delta_G_standard = -n * F * E_standard
    return delta_G_standard

@tool
def calculate_hydrogen_electrode_potential(input_params: str) -> float:
    """
    Calculate the potential of the hydrogen electrode at a given pH.
    
    Parameters:
    - input_params (str): A single string containing all necessary parameters separated by "|" in the following order
        pH
        - pH (float): pH of the solution.
    
    Returns:
    - float: Potential of the hydrogen electrode in V.
    """
    pH = float(input_params.split('|')[0])
    return -0.0591 * pH

@tool
def calculate_log_icorr_i0(input_params: str) -> float:
    """
    Calculate the logarithm of the ratio of corrosion current density to exchange current density.
    
    Parameters:
    - input_params (str): A single string containing all necessary parameters separated by "|" in the following order
        corrosion_current_density|exchange_current_density_hydrogen
        - corrosion_current_density (float): Corrosion current density in A/cm².
        - exchange_current_density_hydrogen (float): Exchange current density of hydrogen on iron surface in A/cm².
    
    Returns:
    - float: Logarithm of the ratio of corrosion current density to exchange current density.
    """
    corrosion_current_density, exchange_current_density_hydrogen = map(float, input_params.split('|'))
    import math
    return math.log10(corrosion_current_density / exchange_current_density_hydrogen)

@tool
def convert_mass_to_grams(input_params: str) -> float:
    """
    Convert mass from kilograms to grams.
    
    Parameters:
    - input_params (str): A single string containing the mass in kilograms separated by "|"
        mass_kg
        - mass_kg (float): Mass in kilograms.
    
    Returns:
    - float: Mass in grams.
    """
    mass_kg = float(input_params.split('|')[0])
    return mass_kg * 1000

@tool
def calculate_moles(input_params: str) -> float:
    """
    Calculate the number of moles given the mass in grams and atomic weight.
    
    Parameters:
    - input_params (str): A single string containing all necessary parameters separated by "|"
        mass_g|atomic_weight_amu
        - mass_g (float): Mass in grams.
        - atomic_weight_amu (float): Atomic weight in atomic mass units (amu).
    
    Returns:
    - float: Number of moles.
    """
    mass_g, atomic_weight_amu = map(float, input_params.split('|'))
    return mass_g / atomic_weight_amu

@tool
def calculate_duration_in_seconds(input_params: str) -> float:
    """
    Calculate the total duration in seconds given the duration in days.
    
    Parameters:
    - input_params (str): A single string containing the duration in days separated by "|"
        duration_days
        - duration_days (float): Duration in days.
    
    Returns:
    - float: Duration in seconds.
    """
    duration_days = float(input_params.split('|')[0])
    return duration_days * 86400

@tool
def calculate_total_charge(input_params: str) -> float:
    """
    Calculate the total charge transferred using Faraday's law.
    
    Parameters:
    - input_params (str): A single string containing all necessary parameters separated by "|"
        moles|n_electrons|Faraday
        - moles (float): Number of moles.
        - n_electrons (int): Number of electrons transferred.
        - Faraday (float): Faraday constant in C/mol.
    
    Returns:
    - float: Total charge in Coulombs.
    """
    moles, n_electrons, Faraday = map(float, input_params.split('|'))
    return moles * n_electrons * Faraday

@tool
def calculate_average_current(input_params: str) -> float:
    """
    Calculate the average current given the total charge and duration.
    
    Parameters:
    - input_params (str): A single string containing all necessary parameters separated by "|"
        total_charge|duration_seconds
        - total_charge (float): Total charge in Coulombs.
        - duration_seconds (float): Duration in seconds.
    
    Returns:
    - float: Average current in Amperes.
    """
    total_charge, duration_seconds = map(float, input_params.split('|'))
    return total_charge / duration_seconds

@tool
def convert_cm_to_m(input_params: str) -> float:
    """
    Convert centimeters to meters.
    
    Parameters:
    - input_params (str): A single string containing the value in centimeters separated by "|":
        value_cm
        - value_cm (float): Value in centimeters.
    
    Returns:
    - float: Value in meters.
    """
    value_cm = float(input_params.split('|')[0])
    return value_cm * 0.01

@tool
def convert_cm2_to_m2(input_params: str) -> float:
    """
    Convert square centimeters to square meters.
    
    Parameters:
    - input_params (str): A single string containing the value in square centimeters separated by "|":
        value_cm2
        - value_cm2 (float): Value in square centimeters.
    
    Returns:
    - float: Value in square meters.
    """
    value_cm2 = float(input_params.split('|')[0])
    return value_cm2 * 0.0001

@tool
def calculate_capacitance(input_params: str) -> float:
    """
    Calculate the capacitance of a parallel plate capacitor with a specified dielectric material.
    
    Parameters:
    - input_params (str): A single string containing all necessary parameters separated by "|":
        dielectric_constant|thickness_m|area_m2
        - dielectric_constant (float): Dielectric constant of the material between the capacitor's plates.
        - thickness_m (float): Thickness of the dielectric layer between the plates in meters.
        - area_m2 (float): Area of one of the capacitor's plates in square meters.
    
    Returns:
    - float: The capacitance of the capacitor in Farads.
    """
    from scipy.constants import epsilon_0
    dielectric_constant, thickness_m, area_m2 = map(float, input_params.split('|'))
    capacitance_F = dielectric_constant * epsilon_0 * area_m2 / thickness_m
    return capacitance_F

@tool
def convert_diameter_to_meters(input_params: str) -> float:
    """
    Convert nozzle diameter from millimeters to meters.

    Parameters:
    - input_params (str): A single string containing the diameter in millimeters separated by a bar.
        - diameter_mm (float): Diameter in millimeters.

    Returns:
    - float: Diameter in meters.
    """
    diameter_mm = float(eval(input_params))
    return diameter_mm / 1000

@tool
def convert_micrometers_to_meters(input_params: str) -> float:
    """
    Convert diameter from micrometers to meters.

    Parameters:
    - input_params (str): A single string containing one parameter separated by "|" in the following order:
        particle_diameter_um
        - particle_diameter_um (float): Diameter in micrometers.

    Returns:
    - float: Diameter in meters.
    """
    particle_diameter_um = float(input_params.split('|')[0])
    return particle_diameter_um * 1e-6


@tool
def calculate_nozzle_area(input_params: str) -> float:
    """
    Calculate the area of the nozzle in square meters.

    Parameters:
    - input_params (str): A single string containing the diameter in meters separated by a bar.
        - diameter_m (float): Diameter of the nozzle in meters.

    Returns:
    - float: Area of the nozzle in square meters.
    """
    import math
    diameter_m = float(eval(input_params))
    return math.pi * (diameter_m / 2) ** 2

@tool
def calculate_velocity(input_params: str) -> float:
    """
    Calculate the velocity of the liquid exiting the nozzle using Torricelli's theorem.

    Parameters:
    - input_params (str): A single string containing the height and gravity separated by a bar.
        - height (float): Height of the liquid column in meters.
        - g (float): Acceleration due to gravity in m/s^2.

    Returns:
    - float: Velocity of the liquid in m/s.
    """
    import math
    height, g = map(eval, input_params.split('|'))
    return math.sqrt(2 * g * height)


@tool
def calculate_granular_bed_volumetric_flow_rate(input_params: str) -> float:
    """
    Calculate the volumetric flow rate per unit area through a granular bed using Darcy's law.

    Parameters:
    - input_params (str): A single string containing the parameters separated by "|" in the following order:
        permeability_m4N1s1|pressure_drop_pa
        - permeability_m4N1s1 (float): Bed permeability in m^4/(N*s).
        - pressure_drop_pa (float): Pressure drop in Pascals (Pa).

    Returns:
    - float: The volumetric flow rate per unit area in m^3/s per m^2.
    """
    permeability_m4N1s1, pressure_drop_pa = map(eval, input_params.split('|'))
    return permeability_m4N1s1 * pressure_drop_pa


@tool
def calculate_liquid_metal_mass_flow_rate(input_params: str) -> float:
    """
    Calculate the mass flow rate of the liquid metal.

    Parameters:
    - input_params (str): A single string containing the density and volumetric flow rate separated by a bar.
        - density (float): Density of the liquid metal in kg/m^3.
        - volumetric_flow_rate (float): Volumetric flow rate in m^3/s.

    Returns:
    - float: Mass flow rate in kg/s.
    """
    density, volumetric_flow_rate = map(eval, input_params.split('|'))
    return density * volumetric_flow_rate


@tool
def convert_pressure_drop_to_pascals(input_params: str) -> float:
    """
    Convert pressure drop from mm of water to Pascals (Pa).

    Parameters:
    - input_params (str): A single string containing the parameters separated by "|" in the following order:
        pressure_drop_mm|density_water_kgm3|gravity_ms2
        - pressure_drop_mm (float): Pressure drop in mm of water.
        - density_water_kgm3 (float): Density of water in kg/m^3.
        - gravity_ms2 (float): Acceleration due to gravity in m/s^2.

    Returns:
    - float: Pressure drop in Pascals.
    """
    pressure_drop_mm, density_water_kgm3, gravity_ms2 = map(eval, input_params.split('|'))
    return density_water_kgm3 * gravity_ms2 * (pressure_drop_mm / 1000)

@tool
def calculate_nozzle_volumetric_flow_rate(input_params: str) -> float:
    """
    Calculate the volumetric flow rate through the nozzle.

    Parameters:
    - input_params (str): A single string containing the discharge coefficient, area, and velocity separated by bars.
        - discharge_coefficient (float): Discharge coefficient of the nozzle.
        - area (float): Area of the nozzle in square meters.
        - velocity (float): Velocity of the liquid in m/s.

    Returns:
    - float: Volumetric flow rate in m^3/s.
    """
    discharge_coefficient, area, velocity = map(eval, input_params.split('|'))
    return discharge_coefficient * area * velocity

@tool
def calculate_free_fall_velocity(input_params: str) -> float:
    """
    Calculate the velocity of an object in free fall.

    Parameters:
    - input_params (str): A single string containing the parameters separated by "|" in the following order:
        gravity|height
        - gravity (float): Acceleration due to gravity in m/s^2.
        - height (float): Height from which the object falls in meters.

    Returns:
    - float: Velocity of the object in m/s.
    """
    gravity, height = map(eval, input_params.split('|'))
    return math.sqrt(2 * gravity * height)

@tool
def calculate_fill_time(input_params: str) -> int:
    """
    Calculate the time required to fill a mold cavity with liquid metal using the given parameters.

    Parameters:
    - input_params (str): A single string containing the parameters separated by "|" in the following order:
        gravity|cross_sectional_area|height|volume
        - gravity (float): Acceleration due to gravity in m/s^2.
        - cross_sectional_area (float): Cross-sectional area of the gate in m^2.
        - height (float): Height from which the liquid metal falls freely in meters.
        - volume (float): Volume of the mold cavity in m^3.

    Returns:
    - int: Time required to fill the mold cavity in seconds, rounded to the nearest integer.
    """
    gravity, cross_sectional_area, height, volume = map(eval, input_params.split('|'))
    velocity = calculate_free_fall_velocity(f'{gravity}|{height}')
    volumetric_flow_rate = cross_sectional_area * velocity
    fill_time = volume / volumetric_flow_rate
    return round(fill_time)

@tool
def calculate_terminal_velocity(input_params: str) -> float:
    """
    Calculate the terminal velocity of a particle using Stokes' law.

    Parameters:
    - input_params (str): A single string containing the parameters separated by "|" in the following order:
        density_steel_kgm3|density_inclusion_kgm3|viscosity_steel_pas|gravity_ms2|diameter_inclusion_m
        - density_steel_kgm3 (float): Density of the steel in kg/m^3.
        - density_inclusion_kgm3 (float): Density of the inclusion in kg/m^3.
        - viscosity_steel_pas (float): Viscosity of the steel in Pascal-seconds (Pa*s).
        - gravity_ms2 (float): Acceleration due to gravity in meters per second squared (m/s^2).
        - diameter_inclusion_m (float): Diameter of the spherical inclusion in meters.

    Returns:
    - float: Terminal velocity in meters per second (m/s).
    """
    rho_steel, rho_inclusion, mu_steel, g, diameter_m = map(eval, input_params.split('|'))
    return (rho_steel - rho_inclusion) * g * diameter_m ** 2 / (18 * mu_steel)

@tool
def calculate_flotation_time(input_params: str) -> float:
    """
    Calculate the time required for a particle to float to the surface.

    Parameters:
    - input_params (str): A single string containing two parameters separated by "|" in the following order:
        depth|velocity
        - depth (float): Depth from which the particle must rise to the surface in meters.
        - velocity (float): Terminal velocity of the particle in m/s.

    Returns:
    - float: Time in seconds.
    """
    params = list(map(float, input_params.split('|')))
    depth, velocity = params
    return depth / velocity

@tool
def convert_velocity_to_mm_per_s(input_params: str) -> float:
    """
    Convert velocity from meters per second (m/s) to millimeters per second (mm/s).

    Parameters:
    - input_params (str): A single string containing the velocity in meters per second separated by a bar.
        - velocity_ms (float): Velocity in meters per second.

    Returns:
    - float: Velocity in millimeters per second.
    """
    velocity_ms = float(eval(input_params))
    return velocity_ms * 1000.0

@tool
def calculate_new_pressure(args: str) -> float:
    """
    Calculate the new pressure of hydrogen.

    Parameters:
        args (str): A single string containing all necessary parameters **two parameters** separated by "|" in the following order:
                    initial_pressure|new_pressure_factor
                    - initial_pressure (float): Initial partial pressure of hydrogen in atm.
                    - new_pressure_factor (float): Factor by which the partial pressure is increased.

    Returns:
        float: The new pressure of hydrogen in atm.

    Example usage:
    >>> calculate_new_pressure("1.0|2.0")
    2.0
    """
    initial_pressure, new_pressure_factor = map(eval, args.split('|'))
    return initial_pressure * new_pressure_factor

@tool
def calculate_solubility_increase_factor(args: str) -> float:
    """
    Calculate the increase factor of solubility of hydrogen.

    Parameters:
        args (str): A single string containing all necessary parameters **one parameter** separated by "|" in the following order:
                    new_pressure_factor
                    - new_pressure_factor (float): Factor by which the partial pressure is increased.

    Returns:
        float: Increase factor of solubility of hydrogen.

    Example usage:
    >>> calculate_solubility_increase_factor("2.0")
    1.4142135623730951
    """
    new_pressure_factor = float(args)
    return math.sqrt(new_pressure_factor)

@tool
def parse_homogenization_parameters(args: str) -> tuple:
    """
    Parse the input parameters for homogenization time calculation.

    Parameters:
        args (str): A single string containing all necessary parameters **three parameters** separated by "|" in the following order:
                    diffusivity_lower_temp|diffusivity_higher_temp|time_lower_temp_hours
                    - diffusivity_lower_temp (float): Diffusivity at the lower temperature in m^2/s.
                    - diffusivity_higher_temp (float): Diffusivity at the higher temperature in m^2/s.
                    - time_lower_temp_hours (float): Time taken to homogenize at the lower temperature in hours.

    Returns:
        tuple: A tuple containing diffusivity_lower_temp (float), diffusivity_higher_temp (float), and time_lower_temp_hours (float).

    Example usage:
    >>> parse_homogenization_parameters("1e-9|2e-9|5")
    (1e-9, 2e-9, 5.0)
    """
    diffusivity_lower_temp, diffusivity_higher_temp, time_lower_temp_hours = map(eval, args.split('|'))
    return (diffusivity_lower_temp, diffusivity_higher_temp, time_lower_temp_hours)

@tool
def calculate_time_higher_temp(args: str) -> float:
    """
    Calculate the time required to achieve the same extent of homogenization at the higher temperature.

    Parameters:
        args (str): A single string containing all necessary parameters **three parameters** separated by "|" in the following order:
                    diffusivity_lower_temp|diffusivity_higher_temp|time_lower_temp_hours
                    - diffusivity_lower_temp (float): Diffusivity at the lower temperature in m^2/s.
                    - diffusivity_higher_temp (float): Diffusivity at the higher temperature in m^2/s.
                    - time_lower_temp_hours (float): Time taken to homogenize at the lower temperature in hours.

    Returns:
        float: Time required to achieve the same extent of homogenization at the higher temperature in hours.

    Example usage:
    >>> calculate_time_higher_temp("1e-9|2e-9|5")
    2.5
    """
    diffusivity_lower_temp, diffusivity_higher_temp, time_lower_temp_hours = map(eval, args.split('|'))
    return diffusivity_lower_temp / diffusivity_higher_temp * time_lower_temp_hours

@tool
def parse_hydrogen_solubility_parameters(args: str) -> tuple:
    """
    Parse the input parameters for hydrogen solubility calculation.

    Parameters:
        args (str): A single string containing all necessary parameters **three parameters** separated by "|" in the following order:
                    solubility_at_ref_pressure_mm3|ref_pressure_atm|new_pressure_atm
                    - solubility_at_ref_pressure_mm3 (float): Solubility of hydrogen in mm^3 (STP) per kg of Pd at the reference pressure.
                    - ref_pressure_atm (float): Reference hydrogen pressure in atmospheres.
                    - new_pressure_atm (float): New hydrogen pressure in atmospheres.

    Returns:
        tuple: A tuple containing solubility_at_ref_pressure_mm3 (float), ref_pressure_atm (float), and new_pressure_atm (float).

    Example usage:
    >>> parse_hydrogen_solubility_parameters("100.0|1.0|2.0")
    (100.0, 1.0, 2.0)
    """
    solubility_at_ref_pressure_mm3, ref_pressure_atm, new_pressure_atm = map(eval, args.split('|'))
    return (solubility_at_ref_pressure_mm3, ref_pressure_atm, new_pressure_atm)

@tool
def calculate_solubility_at_new_pressure(args: str) -> float:
    """
    Calculate the solubility of hydrogen in palladium at a new hydrogen pressure.

    Parameters:
        args (str): A single string containing all necessary parameters **three parameters** separated by "|" in the following order:
                    solubility_at_ref_pressure_mm3|ref_pressure_atm|new_pressure_atm
                    - solubility_at_ref_pressure_mm3 (float): Solubility of hydrogen in mm^3 (STP) per kg of Pd at the reference pressure.
                    - ref_pressure_atm (float): Reference hydrogen pressure in atmospheres.
                    - new_pressure_atm (float): New hydrogen pressure in atmospheres.

    Returns:
        float: Solubility of hydrogen in mm^3 (STP) per kg of Pd at the new pressure.

    Example usage:
    >>> calculate_solubility_at_new_pressure("100.0|1.0|2.0")
    141.4
    """
    solubility_at_ref_pressure_mm3, ref_pressure_atm, new_pressure_atm = map(eval, args.split('|'))
    return solubility_at_ref_pressure_mm3 * (new_pressure_atm / ref_pressure_atm) ** 0.5

@tool
def calculate_time_at_lower_temp(args: str) -> float:
    """
    Calculate the time required to achieve the same concentration profile at a lower temperature.

    Parameters:
        args (str): A single string containing all necessary parameters **three parameters** separated by "|" in the following order:
                    diff_coeff_high_temp|diff_coeff_low_temp|time_high_temp_hours
                    - diff_coeff_high_temp (float): Diffusion coefficient at the higher temperature in m^2/s.
                    - diff_coeff_low_temp (float): Diffusion coefficient at the lower temperature in m^2/s.
                    - time_high_temp_hours (float): Time taken to develop a certain concentration profile at the higher temperature in hours.

    Returns:
        float: Time required to achieve the same concentration profile at the lower temperature in hours.

    Example usage:
    >>> calculate_time_at_lower_temp("1e-9|5e-10|10")
    20.0
    """
    diff_coeff_high_temp, diff_coeff_low_temp, time_high_temp_hours = map(eval, args.split('|'))
    return diff_coeff_high_temp / diff_coeff_low_temp * time_high_temp_hours

@tool
def calculate_sieverts_constant(args: str) -> float:
    """
    Calculate Sieverts' law constant (K_n) for nitrogen solubility in iron.

    Parameters:
        args (str): A single string containing all necessary parameters **one parameter** separated by "|" in the following order:
                    temperature
                    - temperature (float): Temperature in Kelvin.

    Returns:
        float: Sieverts' law constant (K_n).

    Example usage:
    >>> calculate_sieverts_constant("1873")
    2.3773155887399076e-06
    """
    temperature = float(args)
    log_kn = -518 / temperature - 1.063
    k_n = 10 ** log_kn
    return k_n

@tool
def calculate_equilibrium_concentration(args: str) -> float:
    """
    Calculate the equilibrium concentration of dissolved nitrogen in iron.

    Parameters:
        args (str): A single string containing all necessary parameters **two parameters** separated by "|" in the following order:
                    k_n|pressure
                    - k_n (float): Sieverts' law constant.
                    - pressure (float): Pressure in atm.

    Returns:
        float: Equilibrium concentration of dissolved nitrogen in wt.%.

    Example usage:
    >>> calculate_equilibrium_concentration("2.3773155887399076e-06|1")
    2.0857996414621968e-06
    """
    k_n, pressure = map(eval, args.split('|'))
    h_n = k_n * pressure ** 0.5
    return h_n

@tool
def convert_to_weight_percent(args: str) -> float:
    """
    Convert the equilibrium concentration of nitrogen to weight percent.

    Parameters:
        args (str): A single string containing all necessary parameters **one parameter** separated by "|" in the following order:
                    h_n
                    - h_n (float): Equilibrium concentration of dissolved nitrogen in wt.% N.

    Returns:
        float: Equilibrium concentration of dissolved nitrogen in wt.% rounded to three decimal places.

    Example usage:
    >>> convert_to_weight_percent("2.0857996414621968e-06")
    0.0
    """
    h_n = float(args)
    wt_percent_n = h_n / 1.14
    return round(wt_percent_n, 3)

@tool
def convert_units(args: str) -> tuple:
    """
    Convert units from grams to kilograms and cm^2 to m^2.

    Parameters:
        args (str): A single string containing all necessary parameters **two parameters** separated by "|" in the following order:
                    weight_loss_g|surface_area_cm2
                    - weight_loss_g (float): Weight loss in grams.
                    - surface_area_cm2 (float): Surface area in cm^2.

    Returns:
        tuple: A tuple containing weight_loss_kg (float) and surface_area_m2 (float).

    Example usage:
    >>> convert_units("100.0|200.0")
    (0.1, 0.02)
    """
    weight_loss_g, surface_area_cm2 = map(eval, args.split('|'))
    weight_loss_kg = weight_loss_g / 1000
    surface_area_m2 = surface_area_cm2 / 10000
    return (weight_loss_kg, surface_area_m2)

@tool
def calculate_corrosion_rate(args: str) -> float:
    """
    Calculate the corrosion rate given weight loss, surface area, and time period.

    Parameters:
        args (str): A single string containing all necessary parameters **three parameters** separated by "|" in the following order:
                    weight_loss_kg|surface_area_m2|time_years
                    - weight_loss_kg (float): Weight loss in kilograms.
                    - surface_area_m2 (float): Surface area in m^2.
                    - time_years (float): Time period in years.

    Returns:
        float: Corrosion rate in kg/m^2/year.

    Example usage:
    >>> calculate_corrosion_rate("0.1|0.02|2.0")
    2.5
    """
    weight_loss_kg, surface_area_m2, time_years = map(eval, args.split('|'))
    corrosion_rate = weight_loss_kg / (surface_area_m2 * time_years)
    return corrosion_rate

@tool
def convert_celsius_to_kelvin(args: str) -> tuple:
    """
    Convert temperatures from Celsius to Kelvin.

    Parameters:
        args (str): A single string containing all necessary parameters **two parameters** separated by "|" in the following order:
                    T1|T2
                    - T1 (float): Temperature 1 in °C.
                    - T2 (float): Temperature 2 in °C.

    Returns:
        tuple: A tuple containing T1 (float) and T2 (float) in Kelvin.

    Example usage:
    >>> convert_celsius_to_kelvin("25|35")
    (298.15, 308.15)
    """
    T1, T2 = map(float, args.split('|'))
    T1 += 273.15
    T2 += 273.15
    return (T1, T2)

@tool
def calculate_activation_energy_formula(args: str) -> float:
    """
    Calculate the activation energy for diffusion using the given parameters.

    Parameters:
        args (str): A single string containing all necessary parameters **five parameters** separated by "|" in the following order:
                    D1|T1|D2|T2|R
                    - D1 (float): Diffusion coefficient at T1 in m^2/s.
                    - T1 (float): Temperature 1 in Kelvin.
                    - D2 (float): Diffusion coefficient at T2 in m^2/s.
                    - T2 (float): Temperature 2 in Kelvin.
                    - R (float): Universal gas constant in J/(mol·K).

    Returns:
        float: Activation energy in J/mol.

    Example usage:
    >>> calculate_activation_energy_formula("1e-9|298.15|2e-9|308.15|8.314")
    52000.0
    """
    import math
    D1, T1, D2, T2, R = map(eval, args.split('|'))
    Q = -R * math.log(D2 / D1) / (1 / T2 - 1 / T1)
    return Q

@tool
def convert_j_to_kj(args: str) -> float:
    """
    Convert energy from J/mol to kJ/mol.

    Parameters:
        args (str): A single string containing all necessary parameters **one parameter** separated by "|" in the following order:
                    Q
                    - Q (float): Activation energy in J/mol.

    Returns:
        float: Activation energy in kJ/mol.

    Example usage:
    >>> convert_j_to_kj("52000.0")
    52.0
    """
    Q = float(args)
    return Q / 1000

@tool
def calculate_diffusion_time(args: str) -> float:
    """
    Calculate the diffusion time for a given depth, diffusion coefficient, and error function value.

    Parameters:
        args (str): A single string containing all necessary parameters **three parameters** separated by "|" in the following order:
                    x|D|erf_value
                    - x (float): Depth in cm.
                    - D (float): Diffusion coefficient in cm^2/s.
                    - erf_value (float): The value of the error function for the given condition.

    Returns:
        float: Time in seconds.

    Example usage:
    >>> calculate_diffusion_time("0.1|1e-4|0.5")
    5000.0
    """
    x, D, erf_value = map(eval, args.split('|'))
    t = (x / (2 * erf_value)) ** 2 / D
    return t

@tool
def calculate_arrhenius_diffusivity(args: str) -> float:
    """
    Calculate the diffusivity using the Arrhenius equation.

    Parameters:
        args (str): A single string containing all necessary parameters **four parameters** separated by "|" in the following order:
                    D0|Q|R|T_K
                    - D0 (float): Pre-exponential factor in m^2/s.
                    - Q (float): Activation energy in J/mol.
                    - R (float): Universal gas constant in J/(mol·K).
                    - T_K (float): Temperature in Kelvin.

    Returns:
        float: Diffusivity in m^2/s.

    Example usage:
    >>> calculate_arrhenius_diffusivity("1e-4|80000|8.314|1173.15")
    2.780491156370764e-12
    """
    import math
    D0, Q, R, T_K = map(eval, args.split('|'))
    D = D0 * math.exp(-Q / (R * T_K))
    return D

@tool
def convert_diffusivity_units(args: str) -> float:
    """
    Convert diffusivity to units of 10^(-13) m^2/s.

    Parameters:
        args (str): A single string containing all necessary parameters **one parameter** separated by "|" in the following order:
                    D
                    - D (float): Diffusivity in m^2/s.

    Returns:
        float: Diffusivity in units of 10^(-13) m^2/s.

    Example usage:
    >>> convert_diffusivity_units("2.780491156370764e-12")
    27.8
    """
    D = float(args)
    D_in_10_neg_13 = D * 10000000000000.0
    return round(D_in_10_neg_13, 1)

@tool
def calculate_hydrogen_ion_concentration(args: str) -> float:
    """
    Calculate the hydrogen ion concentration from pH.

    Parameters:
        args (str): A single string containing all necessary parameters **one parameter** separated by "-" in the following order:
                    pH
                    - pH (float): pH of the solution.

    Returns:
        float: Hydrogen ion concentration.

    Example usage:
    >>> calculate_hydrogen_ion_concentration("7")
    1e-07
    """
    pH = float(args)
    H_concentration = 10 ** (-pH)
    return H_concentration

@tool
def calculate_nernst_potential_nickel(args: str) -> float:
    """
    Calculate the Nernst potential for nickel.

    Parameters:
        args (str): A single string containing all necessary parameters **four parameters** separated by "-" in the following order:
                    E_Ni_standard|R|T|Ni_concentration
                    - E_Ni_standard (float): Standard reduction potential of nickel in V.
                    - R (float): Universal gas constant in J/(K*mol).
                    - T (float): Temperature in K.
                    - Ni_concentration (float): Concentration of Ni^2+ in M.

    Returns:
        float: Nernst potential for nickel.

    Example usage:
    >>> calculate_nernst_potential_nickel("0.25|8.314|298|0.1")
    0.184
    """
    import math
    E_Ni_standard, R, T, Ni_concentration = map(eval, args.split('-'))
    E_Ni = E_Ni_standard - R * T / (2 * 96485) * math.log(Ni_concentration)
    return E_Ni

@tool
def calculate_partial_pressure_hydrogen(args: str) -> float:
    """
    Calculate the partial pressure of hydrogen.

    Parameters:
        args (str): A single string containing all necessary parameters **five parameters** separated by "-" in the following order:
                    H_concentration|E_H2|E_Ni|R|T
                    - H_concentration (float): Hydrogen ion concentration.
                    - E_H2 (float): Standard hydrogen electrode potential in V.
                    - E_Ni (float): Nernst potential for nickel in V.
                    - R (float): Universal gas constant in J/(K*mol).
                    - T (float): Temperature in K.

    Returns:
        float: Partial pressure of hydrogen in atm.

    Example usage:
    >>> calculate_partial_pressure_hydrogen("1e-07|0|0.184|8.314|298")
    0.085
    """
    import math
    H_concentration, E_H2, E_Ni, R, T = map(eval, args.split('-'))
    P_H2 = 1 / (H_concentration ** 2 * math.exp((E_H2 - E_Ni) / (R * T) * 2 * 96485))
    return P_H2

@tool
def calculate_erf_value(args: str) -> float:
    """
    Calculate the error function value for given concentrations.

    Parameters:
        args (str): A single string containing all necessary parameters **three parameters** separated by "-" in the following order:
                    desired_concentration|surface_concentration|initial_concentration
                    - desired_concentration (float): Desired carbon concentration at the given depth in wt%.
                    - surface_concentration (float): Surface carbon concentration in wt%.
                    - initial_concentration (float): Initial carbon concentration in wt%.

    Returns:
        float: Error function value.

    Example usage:
    >>> calculate_erf_value("0.5|1.0|0.2")
    -0.625
    """
    desired_concentration, surface_concentration, initial_concentration = map(eval, args.split('-'))
    erf_value = (desired_concentration - surface_concentration) / (initial_concentration - surface_concentration)
    return erf_value

@tool
def calculate_inverse_erf(args: str) -> float:
    """
    Calculate the inverse error function value for a given error function value.

    Parameters:
        args (str): A single string containing all necessary parameters **one parameter** separated by "-" in the following order:
                    erf_value
                    - erf_value (float): Error function value.

    Returns:
        float: Inverse error function value.

    Example usage:
    >>> calculate_inverse_erf("-0.625")
    -0.544115
    """
    import math
    erf_value = float(args)
    z = math.erfinv(erf_value)
    return z

@tool
def calculate_carburization_time(args: str) -> float:
    """
    Calculate the carburization time for given parameters.

    Parameters:
        args (str): A single string containing all necessary parameters **four parameters** separated by "-" in the following order:
                    depth|diffusivity|z
                    - depth (float): Depth at which desired concentration is needed in meters.
                    - diffusivity (float): Diffusivity of carbon in iron in m^2/s.
                    - z (float): Inverse error function value.

    Returns:
        float: Carburization time in seconds.

    Example usage:
    >>> calculate_time("0.01|1e-10|-0.544115")
    36000.0
    """
    depth, diffusivity, z = map(eval, args.split('-'))
    t = (depth / (2 * z)) ** 2 / diffusivity
    return t

@tool
def convert_seconds_to_hours(args: str) -> float:
    """
    Convert time from seconds to hours.

    Parameters:
        args (str): A single string containing all necessary parameters **one parameter** separated by "-" in the following order:
                    time_seconds
                    - time_seconds (float): Time in seconds.

    Returns:
        float: Time in hours.

    Example usage:
    >>> convert_seconds_to_hours("36000")
    10.0
    """
    time_seconds = float(args)
    time_hours = time_seconds / 3600
    return time_hours


class MaterialScienceTools:
    tools = [
        convert_temperature_to_kelvin,
        calculate_standard_cell_potential,
        calculate_ln_q,
        calculate_conductivity,
        calculate_free_electron_concentration,
        calculate_atomic_density,
        calculate_potential_vs_she,
        calculate_potential_vs_zn,
        calculate_total_copper_mass,
        calculate_moles_of_copper,
        calculate_total_charge,
        calculate_total_working_hours,
        calculate_conductivity_cm,
        convert_conductivity_to_m,
        calculate_reaction_quotient,
        calculate_nernst_potential,
        convert_voltage_to_millivolts,
        calculate_reaction_quotient_nickel,
        calculate_nernst_voltage,
        convert_voltage_to_millivolts,
        calculate_charge_passed,
        calculate_moles_of_zinc,
        calculate_mass_of_zinc,
        calculate_actual_mass_of_zinc,
        calculate_energy_consumed,
        convert_oxygen_concentration,
        calculate_diffusion_limited_current_density,
        calculate_anodic_current_density,
        calculate_si_atom_density,
        calculate_p_atoms_per_m3,
        calculate_conductivity,
        calculate_delta_g_standard,
        calculate_hydrogen_electrode_potential,
        calculate_log_icorr_i0,
        convert_mass_to_grams,
        calculate_moles,
        calculate_duration_in_seconds,
        calculate_total_charge,
        calculate_average_current,
        convert_cm_to_m,
        convert_cm2_to_m2,
        calculate_capacitance,
        calculate_degrees_of_freedom,
        calculate_max_phases,
        calculate_weight_fractions,
        calculate_total_weight_percent_a,
        calculate_weight_percent_b_in_alloy,
        calculate_mass_fractions_alpha,
        calculate_moles_alpha,
        calculate_mole_fractions_beta,
        calculate_moles_beta,
        calculate_pearlite_fraction,
        calculate_lever_rule_numerator,
        calculate_lever_rule_denominator,
        calculate_rate_constant,
        calculate_recrystallization_percentage,
        convert_pressure_atm_to_pa,
        convert_volume_cm3_to_m3,
        calculate_change_in_volume,
        calculate_change_in_entropy,
        calculate_change_in_pressure,
        convert_pressure_pa_to_atm,
        calculate_bcc_unit_cell_volume,
        calculate_hcp_unit_cell_volume,
        calculate_volume_change_percentage,
        calculate_bcc_unit_cell_volume,
        calculate_fcc_unit_cell_volume,
        calculate_volume_change_percentage,
        calculate_temperature_ratio,
        calculate_transformation_driving_force,
        calculate_driving_force,
        calculate_temperature_ratio,
        calculate_free_energy,
        calculate_carbon_difference,
        calculate_fraction_ferrite,
        convert_diameter_to_meters, 
        convert_micrometers_to_meters,
        calculate_nozzle_area, 
        calculate_velocity, 
        calculate_liquid_metal_mass_flow_rate, 
        calculate_nozzle_volumetric_flow_rate, 
        convert_pressure_drop_to_pascals, 
        calculate_granular_bed_volumetric_flow_rate, 
        calculate_free_fall_velocity, 
        calculate_fill_time, 
        calculate_terminal_velocity,
        calculate_flotation_time, 
        convert_velocity_to_mm_per_s,
        calculate_new_pressure,
        calculate_solubility_increase_factor,
        parse_homogenization_parameters,
        calculate_time_higher_temp,
        parse_hydrogen_solubility_parameters,
        calculate_solubility_at_new_pressure,
        calculate_time_at_lower_temp,
        calculate_sieverts_constant,
        calculate_equilibrium_concentration,
        convert_to_weight_percent,
        convert_units,
        calculate_corrosion_rate,
        convert_celsius_to_kelvin,
        calculate_activation_energy_formula,
        convert_j_to_kj,
        calculate_diffusion_time,
        calculate_arrhenius_diffusivity,
        convert_diffusivity_units,
        calculate_hydrogen_ion_concentration,
        calculate_nernst_potential_nickel,
        calculate_partial_pressure_hydrogen,
        calculate_erf_value,
        calculate_inverse_erf,
        calculate_carburization_time,
        convert_seconds_to_hours
    ]