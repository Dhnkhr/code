student_name = "जगदीप सिंह"
submitted_to = "वैशाली इंगळे"
print("STUDENT NAME:", student_name)
print("SUBMITTED TO:", submitted_to)


KB = {
    "Vehicle Fault Diagnosis": {
        "facts": ["engine_not_starting", "battery_low", "headlights_dim", "starter_clicking",
                  "engine_overheating", "coolant_low", "oil_low", "vehicle_old"],
        "rules": [
            (["battery_low", "headlights_dim"], "electrical_system_weak"),
            (["battery_low", "starter_clicking"], "electrical_system_weak"),
            (["electrical_system_weak", "engine_not_starting"], "starting_system_fault"),
            (["starting_system_fault", "vehicle_old"], "battery_replacement_required"),
            (["engine_overheating", "coolant_low"], "cooling_system_fault"),
            (["engine_overheating", "oil_low"], "lubrication_system_fault"),
            (["cooling_system_fault", "lubrication_system_fault"], "severe_engine_risk"),
        ],
    },
    "Agriculture / Crop Advisory": {
        "facts": ["soil_dry", "temperature_high", "rainfall_low", "irrigation_irregular",
                  "leaves_yellow", "plant_growth_slow", "leaf_edges_brown", "crop_stage_flowering"],
        "rules": [
            (["soil_dry", "temperature_high"], "severe_water_stress"),
            (["rainfall_low", "irrigation_irregular"], "irrigation_deficiency"),
            (["severe_water_stress", "irrigation_deficiency"], "urgent_irrigation_required"),
            (["leaves_yellow", "plant_growth_slow"], "nutrient_problem"),
            (["leaf_edges_brown", "temperature_high"], "heat_stress"),
            (["heat_stress", "severe_water_stress"], "crop_heat_risk"),
            (["crop_stage_flowering", "severe_water_stress"], "flowering_yield_risk"),
        ],
    },
    "Network Troubleshooting": {
        "facts": ["internet_not_working", "router_off", "power_fluctuation", "wifi_not_visible",
                  "cable_disconnected", "ip_not_assigned", "dns_failure", "multiple_devices_affected"],
        "rules": [
            (["router_off", "power_fluctuation"], "router_power_issue"),
            (["router_power_issue", "wifi_not_visible"], "wireless_service_unavailable"),
            (["cable_disconnected", "internet_not_working"], "physical_connection_failure"),
            (["ip_not_assigned", "multiple_devices_affected"], "dhcp_failure"),
            (["dns_failure", "internet_not_working"], "dns_problem"),
            (["dhcp_failure", "dns_problem"], "network_configuration_failure"),
        ],
    },
    "Cybersecurity Alert System": {
        "facts": ["multiple_failed_logins", "unknown_device", "unusual_location", "multiple_ip_addresses",
                  "admin_account", "sensitive_file_access", "large_data_transfer", "password_changed"],
        "rules": [
            (["multiple_failed_logins", "unknown_device"], "abnormal_login"),
            (["unusual_location", "multiple_ip_addresses"], "suspicious_access"),
            (["abnormal_login", "suspicious_access"], "high_risk_login"),
            (["high_risk_login", "admin_account"], "privileged_account_threat"),
            (["sensitive_file_access", "large_data_transfer"], "data_exfiltration"),
            (["privileged_account_threat", "data_exfiltration"], "critical_security_incident"),
            (["password_changed", "multiple_failed_logins"], "account_compromise"),
        ],
    },
}


def show(case):
    print("FACTS:")
    for i, f in enumerate(KB[case]["facts"], 1):
        print(f"  F{i}. {f}")
    print("RULES:")
    for i, (cond, concl) in enumerate(KB[case]["rules"], 1):
        print(f"  R{i}. IF {' AND '.join(cond)} THEN {concl}")

show("Vehicle Fault Diagnosis")


def forward_chaining(facts, rules):
    known = set(facts)
    step = 1
    changed = True
    while changed:
        changed = False
        for cond, concl in rules:
            if all(c in known for c in cond) and concl not in known:
                print(f"Step {step}: IF {' AND '.join(cond)} THEN {concl}")
                known.add(concl)
                changed = True
                step += 1
    return known - set(facts)       # only the newly inferred facts


def backward_chaining(goal, facts, rules, depth=0):
    indent = "  " * depth
    if goal in facts:
        print(f"{indent}{goal} -> known fact")
        return True

    for cond, concl in rules:
        if concl == goal:
            print(f"{indent}Trying rule: IF {' AND '.join(cond)} THEN {goal}")
            if all(backward_chaining(c, facts, rules, depth + 1) for c in cond):
                print(f"{indent}{goal} -> PROVED")
                return True

    print(f"{indent}{goal} -> cannot be proved")
    return False


case = "Vehicle Fault Diagnosis"
facts = ["battery_low", "starter_clicking", "engine_not_starting", "vehicle_old"]
rules = KB[case]["rules"]

print("Given facts:", facts)
print("\n--- FORWARD CHAINING ---")
print("New facts inferred:", forward_chaining(facts, rules))

print("\n--- BACKWARD CHAINING: battery_replacement_required ---")
print("Result:", backward_chaining("battery_replacement_required", facts, rules))

print("\n--- BACKWARD CHAINING: severe_engine_risk ---")
print("Result:", backward_chaining("severe_engine_risk", facts, rules))


cases = list(KB)

while True:
    print("\n===== RULE-BASED EXPERT SYSTEM =====")

    i = 1
    for case_name in cases:
        print(i, case_name)
        i += 1

    print("5. Exit")
    ch = input("Choose case study: ")

    if ch == "5":
        break

    if ch not in ["1", "2", "3", "4"]:
        print("Invalid choice")
        continue

    case = cases[int(ch) - 1]
    show(case)

    ids = input("Enter fact IDs (F1,F2): ").upper().split(",")
    facts = set()

    for fact_id in ids:
        fact_id = fact_id.strip()

        if fact_id.startswith("F") and fact_id[1:].isdigit():
            number = int(fact_id[1:])

            if 1 <= number <= len(KB[case]["facts"]):
                facts.add(KB[case]["facts"][number - 1])

    print("Selected facts:", facts)

    mode = input("1. Forward Chaining  2. Backward Chaining: ")

    if mode == "1":
        result = forward_chaining(facts, KB[case]["rules"])
        print("Inferred:", result if result else "Nothing new")

    elif mode == "2":
        goals = []

        for rule in KB[case]["rules"]:
            conclusion = rule[1]

            if conclusion not in goals:
                goals.append(conclusion)

        i = 1
        for goal in goals:
            print("G" + str(i) + ".", goal)
            i += 1

        g = input("Enter goal ID (G1): ").upper().strip()

        if g.startswith("G") and g[1:].isdigit():
            number = int(g[1:])

            if 1 <= number <= len(goals):
                result = backward_chaining(
                    goals[number - 1], facts, KB[case]["rules"]
                )

                if result:
                    print("VERDICT: PROVED")
                else:
                    print("VERDICT: NOT PROVED")

    else:
        print("Invalid mode")
