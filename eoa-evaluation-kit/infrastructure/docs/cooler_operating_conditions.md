# Hydraulic Cooler System — Operating Conditions & Degradation States

Source: **ZeMA Hydraulic Test Rig Documentation**  
Component: **Secondary Cooling-Filtration Circuit (Heat Exchanger / Fan Unit)**  
Annotated Target: `Cooler condition / %` (Column 1 in `profile.txt`)

---

## 1. Cooler Degradation Levels

The health state of the cooling unit is quantitatively controlled during test cycles into three distinct operational condition states:

| State Code | Condition State | Operational Meaning | Impact on Secondary Circuit |
| :---: | :--- | :--- | :--- |
| **100%** | **Full Efficiency** | Optimal normal operation. The cooler dissipates rated heat capacity. | Normal fluid temperature (`TS2` nominal, `CE` near 100%). |
| **20%** | **Reduced Efficiency** | Moderate thermal degradation. Heat dissipation is throttled or filter is partially obstructed. | Elevated fluid temperature at heat exchanger (`TS2` rises), increased electrical power draw (`EPS1`). |
| **3%** | **Close to Total Failure** | Critical cooling failure. Heat exchanger cannot remove fluid heat. | Severe thermal elevation (`TS2` exceeds safe operating thresholds), risk of fluid breakdown and pump cavitation. |

---

## 2. Physical Interdependencies & Key Drivers

During cooler efficiency degradation:
- **Temperature Elevation (`TS2`)**: As cooler efficiency drops from 100% to 3%, temperature sensor `TS2` (located at the heat exchanger) exhibits significant upward drift.
- **Cooling Efficiency Metric (`CE`)**: The virtual cooling efficiency sensor drops proportionally.
- **Power Consumption (`EPS1`)**: Motor power and pressure lines (`PS1`, `PS5`) reflect changes in hydraulic fluid viscosity as temperature climbs.

---

## 3. Maintenance Guidelines (Condition Monitoring)

- **Normal Operations (100%)**: Standard routine observation. Fluid temperature stable.
- **Warning Threshold (20%)**: Schedule secondary circuit inspection. Verify oil filter cleanliness, check for partial blockage in heat exchanger matrix.
- **Critical Alarm (3%)**: Immediate operational mitigation required. Reduce machine load to prevent thermal breakdown of hydraulic oil and cavitation damage to primary pumps.
