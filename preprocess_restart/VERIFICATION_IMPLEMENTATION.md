# Verification & Sorting Feature Implementation

## Overview

This implementation adds a **Step 7: Verification & Sorting** module to the existing multi-point navigation target selection workflow. The new step uses LLM to verify and rank candidate navigation points, improving the quality and reliability of target selection.

## Key Changes

### 1. Workflow Modification

**Before:**
```
Phase 1: Base Setup (Steps 0,1,2,4)
Phase 2: Multi-point Selection (Steps 3,5,6 x k times) → action.json files
Phase 3: Summary Generation
```

**After:**
```
Phase 1: Base Setup (Steps 0,1,2,4)
Phase 2: Candidate Generation (Steps 3,5,6 x k times) → candidate.json files
Phase 3: Verification & Sorting (Step 7) → verified node list
Phase 4: Final Action Generation → action_iter_*.json files
Phase 5: Summary Generation
```

### 2. New Files Created

1. **`prompts/verify_candidates.txt`** - LLM prompt template for verification
2. **`step_7_verification_sorting.py`** - Main verification module
3. **`action_utils.py`** - Utilities for generating final action files

### 3. Modified Files

1. **`step_6_path_planning.py`** - Now generates `candidate.json` instead of `action.json`
2. **`main_workflow.py`** - Integrated Step 7 and final action generation
3. **`multi_point_utils.py`** - Updated summary format for candidates
4. **`input_config.json`** - Added verification prompt path

## Usage

### Configuration

Add the verification prompt path to your `input_config.json`:

```json
{
  "prompts": {
    "choose_room_prompt": "/path/to/choose_a_room.txt",
    "choose_node_prompt": "/path/to/choose_a_node.txt",
    "verify_candidates_prompt": "/path/to/verify_candidates.txt"
  }
}
```

### Running the Workflow

```bash
cd /home/yaoaa/habitat-lab/preprocess_restart
python main_workflow.py input_config.json
```

## Verification Logic

### Input
- **Verification Image**: Top-down view with all candidate points clearly marked
- **Candidate Indices**: List of node IDs from multi-point summary
- **Goal Object**: Target object to find (e.g., "bed")

### LLM Task
1. **Analysis**: Evaluate each marked point for target object presence/proximity
2. **Verification**: Determine if each point is valid for finding the target
3. **Ranking**: Sort valid points by probability of success

### Output Format
```json
{
  "analysis": {
    "point_123": {"valid": true, "reason": "located directly on bed"},
    "point_64": {"valid": false, "reason": "no furniture visible"}
  },
  "final_ranking": [123, 170]
}
```

## Fallback Strategy

If verification returns an empty list `[]`, the system automatically:
1. Uses the first candidate as fallback (`candidate.json` → `action_iter_1.json`)
2. Logs the fallback usage in the final summary
3. Ensures at least one navigation target is available

## Output Files

### Generated During Verification
- **`verification_candidates.png`** - Image with marked candidate points
- **`verification_log.json`** - Detailed verification process log

### Final Action Files
- **`action_iter_1.json`** - Highest priority verified candidate
- **`action_iter_2.json`** - Second priority verified candidate  
- **`action_summary.json`** - Summary of all generated actions

### Updated Summary
The `multi_point_summary.json` now tracks:
- Candidate files instead of action files
- Verification results
- Final action file mappings

## Error Handling

### Robust Parsing
Multiple parsing strategies for LLM responses:
1. **Direct JSON parsing** (preferred)
2. **JSON fragment extraction** (fallback)
3. **Regex list extraction** (backup)
4. **First candidate fallback** (last resort)

### Error Recovery
- Failed verification → empty list → fallback strategy
- Invalid node IDs → filtered out automatically
- Missing files → clear error messages with workflow continuation

## Benefits

1. **Quality Control**: Reduces false positive selections
2. **Intelligent Prioritization**: Orders candidates by success probability  
3. **Reliability**: Fallback ensures workflow never fails completely
4. **Transparency**: Detailed logs for debugging and analysis
5. **Flexibility**: JSON output format allows easy extension

## Example Workflow

```
Input: k_points = 3, goal_object = "bed"

Phase 1: Base setup → topdown view, wall mask, room segmentation, navigation graph
Phase 2: Generate 3 candidates → candidate.json, candidate_iter_2.json, candidate_iter_3.json
Phase 3: LLM verification → verified_node_ids = [170, 123] (node 64 rejected)
Phase 4: Generate actions → action_iter_1.json (node 170), action_iter_2.json (node 123)
Phase 5: Summary → complete workflow documentation
```

This implementation maintains backward compatibility while significantly improving the quality and reliability of navigation target selection.
