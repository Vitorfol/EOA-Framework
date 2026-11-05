# ZeMA Hydraulic Test Rig — Sensor & Telemetry Specifications

Source: **ZeMA gGmbH / Saarland University** (Saarbrücken, Germany)  
Reference: *Helwig et al., IEEE I2MTC (2015) & Schneider et al., Technisches Messen (2017)*

---

## 1. Test Rig Architecture

The hydraulic test rig consists of two interconnected hydraulic circuits:
1. **Primary Working Circuit**: High-pressure circuit driven by a 250 bar axial piston pump.
2. **Secondary Cooling-Filtration Circuit**: Connected via an oil reservoir tank, responsible for continuous fluid conditioning, filtration, and thermal regulation.

The system cyclically executes repeating **60-second constant load cycles** and continuously samples physical process values.

---

## 2. Sensor Instrumentation & Physical Quantities

All process values are measured by calibrated industrial transducers:

| Sensor Tag | Physical Quantity | Engineering Unit | Sampling Rate | Description |
| :--- | :--- | :--- | :--- | :--- |
| **PS1** | Hydraulic Pressure | `bar` | 100 Hz | Main inlet pressure transducer |
| **PS2** | Hydraulic Pressure | `bar` | 100 Hz | Secondary working pressure |
| **PS3** | Hydraulic Pressure | `bar` | 100 Hz | Pump differential pressure |
| **PS4** | Hydraulic Pressure | `bar` | 100 Hz | Return line pressure |
| **PS5** | Hydraulic Pressure | `bar` | 100 Hz | Secondary circuit filtration pressure |
| **PS6** | Hydraulic Pressure | `bar` | 100 Hz | Oil reservoir tank pressure |
| **EPS1** | Motor Power | `W` / `kW` | 100 Hz | Electrical power consumption of the main pump |
| **FS1** | Volume Flow | `l/min` | 10 Hz | Volume flow rate in primary circuit |
| **FS2** | Volume Flow | `l/min` | 10 Hz | Volume flow rate in cooling-filtration circuit |
| **TS1** | Temperature | `°C` | 1 Hz | Fluid temperature at pump suction |
| **TS2** | Temperature | `°C` | 1 Hz | Fluid temperature at cooler (heat exchanger) |
| **TS3** | Temperature | `°C` | 1 Hz | Fluid temperature at return line |
| **TS4** | Temperature | `°C` | 1 Hz | Ambient reservoir temperature |
| **VS1** | Vibration | `mm/s` | 1 Hz | Mechanical vibration velocity |
| **CE** | Cooling Efficiency | `%` | 1 Hz | Virtual sensor: actual vs. nominal cooling efficiency |
| **CP** | Cooling Power | `kW` | 1 Hz | Virtual sensor: thermal power removed by heat exchanger |
| **SE** | Efficiency Factor | `%` | 1 Hz | Virtual sensor: global hydraulic-to-electric efficiency |

---

## 3. Operational Standards & Cycle Profiles

- **Cycle Duration**: Exactly 60 seconds per cycle.
- **Physical Sampling**:
  - High-speed sensors (100 Hz): 6,000 points per cycle.
  - Medium-speed sensors (10 Hz): 600 points per cycle.
  - Thermal / low-speed sensors (1 Hz): 60 points per cycle.
