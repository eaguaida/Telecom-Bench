# Markdown-KV Transformation for Telecom Prompts

## Overview

This feature transforms pipe-delimited and CSV telecom evaluation prompts into **Markdown-KV (Key-Value) format** for optimal LLM comprehension.

### Performance Impact

Markdown-KV format achieves approximately **~60% accuracy** compared to **~41-44%** for pipe-delimited/CSV formats in LLM table understanding benchmarks. The format explicitly labels each field, significantly improving parsing reliability for complex telecom data.

## Rationale

Traditional pipe-delimited tables are difficult for LLMs to parse reliably because:
- Column headers are separated from data by many rows
- Multiple pipes create ambiguous boundaries
- Dense formatting makes relationships unclear
- Null/empty values are represented inconsistently

Markdown-KV format solves these issues by:
- Explicitly labeling each field with its key
- Using clear section headers for data organization
- Grouping related data in code blocks
- Omitting null values entirely
- Aggregating repeated patterns (like neighbor cells) into lists

## Usage

### Option 1: Extract with Transformation

When extracting the TeleLogs dataset, use the `--markdown-kv` flag:

```bash
python extract_telelogs.py --markdown-kv
```

This will:
1. Download the dataset from HuggingFace
2. Extract and clean the data
3. Transform questions to Markdown-KV format
4. Save to `telelogs/telelogs_test_mkv.json` (and .csv, .parquet)

### Option 2: Transform Existing Dataset

If you already have the dataset extracted, transform it using:

```bash
# Transform the default dataset
python transform_to_markdown_kv.py

# Transform and upload to HuggingFace
python transform_to_markdown_kv.py --upload

# Transform with custom HuggingFace repository
python transform_to_markdown_kv.py --upload --repo-id myuser/my-telelogs

# Transform specific input/output
python transform_to_markdown_kv.py --input data.json --output data_mkv.json

# Show sample comparison
python transform_to_markdown_kv.py --sample 0

# Skip validation (faster)
python transform_to_markdown_kv.py --no-validate
```

**Note**: To upload to HuggingFace, make sure you have the `HF_TOKEN` environment variable set:

```bash
export HF_TOKEN=your_huggingface_token_here
```

### Option 3: Use in Your Code

```python
from telecom_bench.prompt_transformer import transform_prompt

# Your pipe-delimited prompt
original_prompt = """
Given:
- Rules and parameters...

User plane drive test data as follows：
Timestamp|Longitude|Latitude|...
2025-05-07 10:25:34|128.139682|32.623035|...

Engineering parameters data as follows：
gNodeB ID|Cell ID|...
0000258|1|...
"""

# Transform to Markdown-KV
transformed = transform_prompt(original_prompt)
print(transformed)
```

## Transformation Examples

### Before (Pipe-Delimited)

```
Given:
- The default electronic downtilt value is 255, representing a downtilt angle of 6 degrees.

User plane drive test data as follows：

Timestamp|Longitude|Latitude|GPS Speed (km/h)|5G KPI PCell RF Serving PCI|5G KPI PCell RF Serving SS-RSRP [dBm]
2025-05-07 10:25:34.000000|128.139682|32.623035|34|919|-85.48

Engineering parameters data as follows：

gNodeB ID|Cell ID|Longitude|Latitude|PCI
0000258|1|128.139529|32.623035|737
```

### After (Markdown-KV)

```markdown
# Domain Rules

## Downtilt Interpretation
- Value 255 → 6° (default)
- Other values → actual degrees

# Drive Test Data

## Record 1 (2025-05-07 10:25:34.000000)
```
timestamp: 2025-05-07 10:25:34.000000
lon: 128.139682
lat: 32.623035
speed_kmh: 34
serving_pci: 919
ss_rsrp_dbm: -85.48
```

# Engineering Parameters

## Cell PCI=737
```
gnodeb_id: 0000258
cell_id: 1
lon: 128.139529
lat: 32.623035
pci: 737
```

# Data Relationships
- Drive test `serving_pci` and `neighbor_pcis` link to Engineering Parameters via `PCI`
```

## Transformation Rules

### 1. Document Structure

```markdown
# Domain Rules
[prose explanation of domain-specific rules]

# Drive Test Data
## Record N
```
key: value
```

# Engineering Parameters
## Cell PCI=X
```
key: value
```

# Data Relationships
```

### 2. Column Name Abbreviations

Long, verbose column names are transformed to concise keys:

| Original | Abbreviated |
|----------|-------------|
| `5G KPI PCell RF Serving SS-RSRP [dBm]` | `ss_rsrp_dbm` |
| `5G KPI PCell RF Serving SS-SINR [dB]` | `ss_sinr_db` |
| `5G KPI PCell Layer2 MAC DL Throughput [Mbps]` | `dl_tput_mbps` |
| `GPS Speed (km/h)` | `speed_kmh` |
| `Longitude` | `lon` |
| `Latitude` | `lat` |
| `Mechanical Azimuth` | `azimuth_mech` |
| `Digital Tilt` | `downtilt_digital` |
| `Beam Scenario` | `beam_scenario` |
| `gNodeB ID` | `gnodeb_id` |

### 3. Value Handling

- **Null/Empty values**: Omitted entirely (no `key: -` or `key: null`)
- **Lists**: Bracket notation `[val1, val2, val3]`
- **Units**: Appended to key name with underscore: `height_m`, `speed_kmh`
- **Numeric precision**: Preserved from original
- **Timestamps**: ISO format `YYYY-MM-DD HH:MM:SS`

### 4. Neighbor Cell Aggregation

Multiple neighbor cell columns are collapsed into lists:

**Before:**
```
Top 1 PCI: 737
Top 2 PCI: 36
Top 3 PCI: 430
Top 1 BRSRP: -88.89
Top 2 BRSRP: -100.49
Top 3 BRSRP: -107.53
```

**After:**
```
neighbor_pcis: [737, 36, 430]
neighbor_rsrp_dbm: [-88.89, -100.49, -107.53]
```

## Validation

The transformation includes automatic validation checks:

- ✓ `has_domain_rules`: Domain rules section present
- ✓ `has_drive_test`: Drive test data section present
- ✓ `has_engineering_params`: Engineering parameters section present
- ✓ `has_relationships`: Data relationships declared
- ✓ `no_pipe_delimiters`: No pipe-delimited tables remain
- ✓ `has_code_blocks`: Data properly formatted in code blocks
- ✓ `no_null_values`: No null/empty value markers

Run validation on a sample:

```bash
python transform_to_markdown_kv.py --sample 0
```

## Testing

Test the transformation on sample data:

```bash
python test_transformation.py
```

This will:
1. Load sample telecom prompt data
2. Transform from pipe-delimited to Markdown-KV
3. Display both formats side-by-side
4. Run all validation checks
5. Report pass/fail status

## Architecture

### Module: `src/telecom_bench/prompt_transformer.py`

Core transformation logic with the following key functions:

- `ColumnMapping.transform(col_name)`: Maps verbose column names to abbreviated keys
- `parse_pipe_delimited(text)`: Parses pipe-delimited tables into headers and rows
- `aggregate_neighbors(record)`: Collapses neighbor cells into lists
- `record_to_markdown_kv(record, record_id)`: Converts a record to Markdown-KV block
- `transform_drive_test_table(table_text)`: Transforms drive test data
- `transform_engineering_params(table_text)`: Transforms engineering parameters
- `extract_domain_rules(text)`: Extracts and reformats domain rules
- `transform_prompt(prompt)`: Full transformation pipeline
- `validate_transformation(original, transformed)`: Validates quality

### Script: `transform_to_markdown_kv.py`

Command-line tool for batch transformation of JSON datasets.

### Script: `test_transformation.py`

Standalone test script with sample data.

## Performance Considerations

### Processing Time

- **Small datasets** (<1000 samples): ~5-10 seconds
- **Medium datasets** (1000-10000 samples): ~30-60 seconds
- **Large datasets** (>10000 samples): ~2-5 minutes

Use `--no-validate` flag to skip validation checks for faster processing.

### Memory Usage

The transformation processes datasets in-memory using pandas DataFrames. For very large datasets (>100K samples), consider processing in chunks.

## Integration with Evaluation Pipeline

To use Markdown-KV format in your evaluations:

### Update Dataset Loading

```python
from telecom_bench.scripts.utils import load_huggingface_dataset
from inspect_ai.dataset import FieldSpec

# Load Markdown-KV formatted dataset
dataset = load_huggingface_dataset(
    "eaguaida/telelogs-mkv",  # or use local path
    sample_fields=FieldSpec(
        input="question",
        target="answer",
    )
)
```

### Update Prompts

The Markdown-KV format works better with minimal additional instructions:

```python
INSTRUCTIONS = dedent(
    """
    You are analyzing 5G network data to identify root causes of issues.
    The data is provided in structured Markdown format with clearly labeled fields.
    Review the domain rules, examine the drive test measurements, and consult
    the engineering parameters to determine the most likely cause.

    Submit your answer as \\boxed{C<number>} where <number> is the cause identifier.
    """
)
```

## Benefits Summary

1. **Higher Accuracy**: ~60% vs ~41-44% on table understanding benchmarks
2. **Better Parsing**: Explicit labels reduce ambiguity
3. **Cleaner Data**: Null values omitted, reducing noise
4. **Structured Organization**: Clear sections and hierarchy
5. **Aggregated Patterns**: Lists for repeated data (neighbors)
6. **Explicit Relationships**: Join keys clearly stated
7. **Human Readable**: Easier to debug and inspect

## Limitations

1. **File Size**: Markdown-KV format is more verbose (~20-30% larger)
2. **Legacy Tools**: Some tools expect CSV/pipe-delimited format
3. **Transformation Time**: Additional processing step required
4. **Custom Rules**: Domain rules extraction may need customization per dataset

## Future Enhancements

Potential improvements:

- [ ] Support for additional table formats (TSV, Excel)
- [ ] Configurable column mappings via config file
- [ ] Streaming transformation for very large datasets
- [ ] Integration with other telecom benchmarks (TeleQnA, TeleMath)
- [ ] Automatic detection of optimal format per model
- [ ] Compression strategies for reduced file size

## Contributing

To extend the transformation rules:

1. Edit column mappings in `ColumnMapping.PATTERNS`
2. Update domain rules extraction logic in `extract_domain_rules()`
3. Add new validation checks to `validate_transformation()`
4. Test with `python test_transformation.py`
5. Update documentation

## References

- [Original TeleLogs Dataset](https://huggingface.co/datasets/netop/TeleLogs)
- [TeleLogs Paper](https://arxiv.org/abs/2507.21974)
- [Telecom-Bench Repository](https://github.com/eaguaida/Telecom-Bench)

## Support

For issues or questions:
- Open an issue at [Telecom-Bench GitHub](https://github.com/eaguaida/Telecom-Bench/issues)
- Check the validation results: `python transform_to_markdown_kv.py --sample <index>`
- Review test output: `python test_transformation.py`
