#!/usr/bin/env python3
"""Test the extraction and transformation process with mock data."""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from extract_telelogs import (
    extract_choices_from_question,
    extract_actual_question,
    transform_answer
)
from telecom_bench.prompt_transformer import transform_prompt

# Mock dataset format (simulating the HuggingFace TeleLogs format)
# This includes the template with C1-C8 choices followed by the actual question
MOCK_QUESTION = """You are a telecom network optimization expert. Analyze the following data and select the most likely cause of poor user-plane throughput.

C1: The serving cell's downtilt angle is too large, causing weak coverage at the far end.
C2: The serving cell's coverage distance exceeds 1km, resulting in over-shooting.
C3: A neighboring cell provides higher throughput.
C4: Non-colocated co-frequency neighboring cells cause severe overlapping coverage.
C5: Frequent handovers degrade performance.
C6: Neighbor cell and serving cell have the same PCI mod 30, leading to interference.
C7: Test vehicle speed exceeds 40km/h, impacting user throughput.
C8: Average scheduled RBs are below 160, affecting throughput.

Given:
- The default electronic downtilt value is 255, representing a downtilt angle of 6 degrees. Other values represent the actual downtilt angle in degrees.

Beam Scenario and Vertical Beamwidth Relationships:
- When the cell's Beam Scenario is set to Default or SCENARIO_1 to SCENARIO_5, the vertical beamwidth is 6 degrees.
- When the Beam Scenario is SCENARIO_6 to SCENARIO_11, the vertical beamwidth is 12 degrees.
- When the Beam Scenario is SCENARIO_12 or higher, the vertical beamwidth is 25 degrees.

User plane drive test data as follows：

Timestamp|Longitude|Latitude|GPS Speed (km/h)|5G KPI PCell RF Serving PCI|5G KPI PCell RF Serving SS-RSRP [dBm]|5G KPI PCell RF Serving SS-SINR [dB]|5G KPI PCell Layer2 MAC DL Throughput [Mbps]|5G KPI PCell Layer1 DL RB Num (Including 0)|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 Filtered Tx BRSRP [dBm]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 2 Filtered Tx BRSRP [dBm]|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 PCI|Measurement PCell Neighbor Cell Top Set(Cell Level) Top 3 Filtered Tx BRSRP [dBm]
2025-05-07 10:25:34.000000|128.139682|32.623035|34|919|-85.48|-7.88|576.94|80|737|-88.89|36|-100.49|430|-107.53
2025-05-07 10:25:35.000000|128.139685|32.623038|35|919|-86.12|-8.15|542.31|75|737|-89.21|36|-101.03|430|-108.11

Engineering parameters data as follows：

gNodeB ID|Cell ID|Longitude|Latitude|Mechanical Azimuth|Mechanical Downtilt|Digital Tilt|Digital Azimuth|Max Transmit Power|TxRx Mode|Beam Scenario|Antenna Model|Height|PCI
0000258|1|128.139529|32.623035|45|3|7|5|34.9|32T32R|SCENARIO_7|NR AAU 1|9.0|737
0000258|2|128.139529|32.623035|165|3|7|5|34.9|32T32R|SCENARIO_7|NR AAU 1|9.0|36
0000258|3|128.139529|32.623035|285|3|7|5|34.9|32T32R|SCENARIO_7|NR AAU 1|9.0|430
0000259|1|128.142335|32.625182|60|2|255|0|34.9|32T32R|SCENARIO_6|NR AAU 1|10.0|919
"""

MOCK_ANSWER = "C1"


def test_extraction_pipeline():
    """Test the full extraction pipeline."""
    print("=" * 80)
    print("TESTING EXTRACTION PIPELINE")
    print("=" * 80)

    print("\n--- Step 1: Extract Choices ---")
    choices = extract_choices_from_question(MOCK_QUESTION)
    print(f"Extracted {len(choices)} choices:")
    for i, choice in enumerate(choices, 1):
        print(f"  C{i}: {choice[:60]}...")

    print("\n--- Step 2: Extract Actual Question ---")
    actual_question = extract_actual_question(MOCK_QUESTION)
    print(f"Extracted question length: {len(actual_question)} chars")
    print(f"First 300 chars:\n{actual_question[:300]}...")

    # Check what we extracted
    has_given = 'Given:' in actual_question
    has_drive_test = 'drive test' in actual_question.lower()
    has_eng_params = 'engineering' in actual_question.lower()

    print(f"\nExtraction validation:")
    print(f"  ✓ Contains 'Given:': {has_given}")
    print(f"  ✓ Contains drive test: {has_drive_test}")
    print(f"  ✓ Contains engineering: {has_eng_params}")

    if not (has_given and has_drive_test and has_eng_params):
        print("\n❌ EXTRACTION FAILED!")
        print(f"Full extracted question:\n{actual_question}")
        return False

    print("\n--- Step 3: Transform Answer ---")
    transformed_answer = transform_answer(MOCK_ANSWER)
    print(f"Original: {MOCK_ANSWER} → Transformed: {transformed_answer}")

    print("\n--- Step 4: Apply Markdown-KV Transformation ---")
    transformed_question = transform_prompt(actual_question)
    print(f"Transformed question length: {len(transformed_question)} chars")

    # Validate transformation
    checks = {
        'has_domain_rules': '# Domain Rules' in transformed_question,
        'has_drive_test': '# Drive Test Data' in transformed_question,
        'has_engineering': '# Engineering Parameters' in transformed_question,
        'has_relationships': '# Data Relationships' in transformed_question,
        'has_code_blocks': '```' in transformed_question
    }

    print("\nTransformation validation:")
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    print("\n--- Step 5: Create Final JSON Record ---")
    record = {
        "question": transformed_question,
        "answer": transformed_answer,
        "choices": choices
    }

    # Show first 500 chars of question field
    print(f"Final question field (first 500 chars):")
    print(record["question"][:500])
    print("...")

    print("\n--- Step 6: Validate JSON Structure ---")
    # Test JSON serialization
    try:
        json_str = json.dumps([record], indent=2)
        print(f"✓ JSON serialization successful")
        print(f"✓ Total JSON size: {len(json_str)} bytes")

        # Parse it back to verify
        parsed = json.loads(json_str)
        print(f"✓ JSON parsing successful")
        print(f"✓ Record count: {len(parsed)}")

    except Exception as e:
        print(f"✗ JSON error: {e}")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("The extraction and transformation pipeline is working correctly.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please review the output above for details.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = test_extraction_pipeline()
    sys.exit(0 if success else 1)
