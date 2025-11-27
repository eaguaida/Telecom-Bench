#!/usr/bin/env python3
"""Test the telecom prompt transformer with sample data."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from telecom_bench.prompt_transformer import transform_prompt, validate_transformation


# Sample pipe-delimited prompt (typical TeleLogs format)
SAMPLE_PROMPT = """Given:
- The default electronic downtilt value is 255, representing a downtilt angle of 6 degrees. Other values represent the actual downtilt angle in degrees.

Beam Scenario and Vertical Beamwidth Relationships:
- When the cell's Beam Scenario is set to Default or SCENARIO_1 to SCENARIO_5, the vertical beamwidth is 6 degrees.
- When the Beam Scenario is SCENARIO_6 to SCENARIO_11, the vertical beamwidth is 12 degrees.
- When the Beam Scenario is SCENARIO_12 or higher, the vertical beamwidth is 25 degrees.

User plane drive test data as follows：

Timestamp|Longitude|Latitude|GPS Speed (km/h)|5G KPI PCell RF Serving PCI|5G KPI PCell RF Serving SS-RSRP [dBm]|5G KPI PCell RF Serving SS-SINR [dB]|5G KPI PCell Layer2 MAC DL Throughput [Mbps]|5G KPI PCell Layer1 DL RB Num (Including 0)|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 Filtered Tx BRSRP [dBm]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 Filtered Tx BRSRP [dBm]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 Filtered Tx BRSRP [dBm]
2025-05-07 10:25:34.000000|128.139682|32.623035|34|919|-85.48|-7.88|576.94|80|737|-88.89|36|-100.49|430|-107.53

Engineering parameters data as follows：

gNodeB ID|Cell ID|Longitude|Latitude|Mechanical Azimuth|Mechanical Downtilt|Digital Tilt|Digital Azimuth|Max Transmit Power|TxRx Mode|Beam Scenario|Antenna Model|Height|PCI
0000258|1|128.139529|32.623035|45|3|7|5|34.9|32T32R|SCENARIO_7|NR AAU 1|9.0|737
0000258|2|128.139529|32.623035|165|3|7|5|34.9|32T32R|SCENARIO_7|NR AAU 1|9.0|36
0000258|3|128.139529|32.623035|285|3|7|5|34.9|32T32R|SCENARIO_7|NR AAU 1|9.0|430
0000259|1|128.142335|32.625182|60|2|255|0|34.9|32T32R|SCENARIO_6|NR AAU 1|10.0|919
"""


def test_transformation():
    """Test the transformation on sample data."""
    print("=" * 80)
    print("TESTING TELECOM PROMPT TRANSFORMATION")
    print("=" * 80)

    print("\n--- ORIGINAL PROMPT (Pipe-Delimited) ---")
    print(SAMPLE_PROMPT)

    print("\n--- TRANSFORMING... ---")
    transformed = transform_prompt(SAMPLE_PROMPT)

    print("\n--- TRANSFORMED PROMPT (Markdown-KV) ---")
    print(transformed)

    print("\n--- VALIDATION RESULTS ---")
    checks = validate_transformation(SAMPLE_PROMPT, transformed)
    all_passed = True

    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED!")
    else:
        print("⚠ SOME VALIDATION CHECKS FAILED")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = test_transformation()
    sys.exit(0 if success else 1)
