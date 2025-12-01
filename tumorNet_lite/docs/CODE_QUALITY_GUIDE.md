# Code Quality Enhancement Guide - TumorNet-Lite

## 🛡️ Comprehensive Error Handling & Validation

This document describes the error handling and code quality improvements added to the TumorNetLitev2.ipynb notebook.

---

## Table of Contents

1. [Error Handling Framework](#error-handling-framework)
2. [Custom Exceptions](#custom-exceptions)
3. [Validation Functions](#validation-functions)
4. [Safe Operation Wrappers](#safe-operation-wrappers)
5. [Integration Examples](#integration-examples)
6. [Best Practices](#best-practices)
7. [Testing](#testing)

---

## Error Handling Framework

### Context Manager: `error_handler()`

**Purpose**: Wrap risky operations with comprehensive error handling.

**Signature**:
```python
@contextmanager
def error_handler(operation_name: str, raise_error: bool = False, 
                 fallback_value: Any = None)
```

**Parameters**:
- `operation_name`: Descriptive name of the operation
- `raise_error`: Whether to re-raise exception after logging (default: False)
- `fallback_value`: Value to return if operation fails (default: None)

**Usage Examples**:
```python
# Example 1: Non-critical operation with fallback
with error_handler("Loading optional config", fallback_value={}):
    config = load_config('config.json')

# Example 2: Critical operation that must succeed
with error_handler("Loading model", raise_error=True):
    model = torch.load('model.pth')

# Example 3: Data loading with default
with error_handler("Loading preprocessed data", fallback_value=None):
    cached_data = np.load('cache.npy')
```

**Features**:
- ✅ Catches all exceptions except KeyboardInterrupt
- ✅ Logs exception type, message, and full traceback
- ✅ Optionally re-raises for critical operations
- ✅ Provides fallback values for graceful degradation
- ✅ Colored output for easy identification (❌ for errors, ⚠️ for warnings)

---

## Custom Exceptions

### Base Exception: `NotebookError`

```python
class NotebookError(Exception):
    """Base exception class for notebook-specific errors"""
    pass
```

### Specialized Exceptions

#### 1. `DataValidationError`
**Purpose**: Raised when data validation fails (shape, type, range checks).

**When to use**:
- Data shape mismatches
- Invalid data types
- Out-of-range values
- Empty datasets

**Example**:
```python
if len(y_true) != len(y_pred):
    raise DataValidationError(
        f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
    )
```

#### 2. `ModelError`
**Purpose**: Raised when model operations fail.

**When to use**:
- Model output contains NaN/Inf
- Unexpected output shapes
- Model loading failures
- Training instabilities

**Example**:
```python
if torch.isnan(output).any():
    raise ModelError("Model output contains NaN values")
```

#### 3. `VisualizationError`
**Purpose**: Raised when visualization operations fail.

**When to use**:
- Figure creation errors
- Plot saving failures
- Invalid visualization parameters

**Example**:
```python
if not safe_save_figure('plot.png'):
    raise VisualizationError("Failed to save visualization")
```

---

## Validation Functions

### 1. `validate_data_shapes()`

**Purpose**: Validate array shapes before processing.

**Signature**:
```python
def validate_data_shapes(data_dict: dict, expected_shapes: dict, 
                         operation: str = "data processing") -> bool
```

**Parameters**:
- `data_dict`: Dictionary mapping names to data arrays
- `expected_shapes`: Dictionary mapping names to expected shapes (use `None` for flexible dimensions)
- `operation`: Description of operation for error messages

**Returns**: `True` if validation passes

**Raises**: `DataValidationError` if validation fails

**Example**:
```python
# Validate training data
validate_data_shapes(
    data_dict={
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val
    },
    expected_shapes={
        'x_train': (None, 200, 200, 3),  # None = any size OK
        'y_train': (None,),
        'x_val': (None, 200, 200, 3),
        'y_val': (None,)
    },
    operation="training data preparation"
)
```

**Features**:
- ✅ Supports flexible dimensions (use `None`)
- ✅ Clear error messages showing expected vs actual
- ✅ Checks for shape attribute existence
- ✅ Success confirmation message

---

### 2. `validate_model_output()`

**Purpose**: Validate model predictions for correctness.

**Signature**:
```python
def validate_model_output(output: torch.Tensor, expected_shape: tuple, 
                         operation: str = "model inference") -> bool
```

**Parameters**:
- `output`: Tensor from model
- `expected_shape`: Expected shape (use `None` for flexible dimensions)
- `operation`: Operation description

**Returns**: `True` if validation passes

**Raises**: `ModelError` if validation fails

**Checks**:
- ✅ Output is a tensor
- ✅ Shape matches expectation
- ✅ No NaN values
- ✅ No Inf values

**Example**:
```python
# After model forward pass
output = model(input_tensor)
validate_model_output(
    output,
    expected_shape=(batch_size, num_classes),
    operation="batch inference"
)
```

---

### 3. `check_dependencies()`

**Purpose**: Verify all required packages are installed.

**Signature**:
```python
def check_dependencies() -> dict
```

**Returns**: Dictionary with package availability status

**Checked Packages**:
- torch, torchvision
- numpy, pandas
- matplotlib, seaborn
- sklearn, scipy
- cv2 (opencv)
- PIL (Pillow)
- tqdm

**Example Output**:
```
======================================================================
DEPENDENCY CHECK
======================================================================
✓ torch          : Available
✓ torchvision    : Available
✓ numpy          : Available
✓ pandas         : Available
✓ matplotlib     : Available
✓ seaborn        : Available
✓ sklearn        : Available
✗ cv2            : MISSING
✓ PIL            : Available
✓ tqdm           : Available

⚠️  Some dependencies missing - install with:
   pip install -r requirements.txt
======================================================================
```

**Usage**:
```python
# Run at notebook start
deps = check_dependencies()

# Check specific package
if not deps['cv2']:
    print("OpenCV not available, using alternative...")
```

---

### 4. `memory_check()`

**Purpose**: Monitor available GPU/RAM memory.

**Signature**:
```python
def memory_check(device: torch.device) -> dict
```

**Parameters**:
- `device`: PyTorch device to check

**Returns**: Dictionary with memory statistics

**Example Output (GPU)**:
```
======================================================================
GPU MEMORY STATUS
======================================================================
Device: NVIDIA GeForce RTX 3080
Allocated: 2.45 GB
Reserved:  3.12 GB
Total:     10.00 GB
Free:      7.55 GB
======================================================================
```

**Example Output (CPU)**:
```
======================================================================
SYSTEM MEMORY STATUS
======================================================================
Total:     16.00 GB
Used:      8.34 GB
Available: 7.66 GB
Usage:     52.1%
======================================================================
```

**Usage**:
```python
# Check before large operation
mem_info = memory_check(device)

if mem_info['free_gb'] < 2.0:
    print("Low GPU memory - reducing batch size")
    batch_size = batch_size // 2
```

---

## Safe Operation Wrappers

### 1. `safe_gpu_operation()`

**Purpose**: Execute GPU operations with automatic CPU fallback.

**Signature**:
```python
def safe_gpu_operation(func: Callable, *args, fallback_device: str = 'cpu', 
                       **kwargs) -> Any
```

**Parameters**:
- `func`: Function to execute
- `*args`: Function arguments
- `fallback_device`: Device for fallback (default: 'cpu')
- `**kwargs`: Function keyword arguments

**Returns**: Result of function execution

**Features**:
- ✅ Catches CUDA out-of-memory errors
- ✅ Automatic GPU cache clearing
- ✅ Moves tensors to fallback device
- ✅ Seamless execution continuation

**Example**:
```python
# Safe model inference
output = safe_gpu_operation(
    lambda: model(input_tensor.to('cuda')),
    fallback_device='cpu'
)

# Safe batch processing
results = safe_gpu_operation(
    process_batch,
    batch_data,
    model=model,
    fallback_device='cpu'
)
```

---

### 2. `safe_save_figure()`

**Purpose**: Robustly save matplotlib figures.

**Signature**:
```python
def safe_save_figure(filename: str, dpi: int = 300, 
                    bbox_inches: str = 'tight') -> bool
```

**Parameters**:
- `filename`: Output file path
- `dpi`: Resolution (default: 300 for publication)
- `bbox_inches`: Bounding box (default: 'tight')

**Returns**: `True` if successful, `False` otherwise

**Features**:
- ✅ Handles file permission errors
- ✅ Creates directory if needed
- ✅ Confirmation message on success
- ✅ Error message on failure (non-blocking)

**Example**:
```python
# Create plot
plt.figure(figsize=(10, 8))
plt.plot(x, y)
plt.title('Results')

# Safe save
if safe_save_figure('results.png', dpi=300):
    print("Plot saved successfully")
else:
    print("Could not save plot, continuing...")
```

---

## Integration Examples

### Example 1: Safe Model Loading

```python
def safe_load_model(model_path, model_class, device, fallback_pretrained=True):
    """Load model with fallback to new model if loading fails."""
    with error_handler("Loading model checkpoint", raise_error=False):
        if os.path.exists(model_path):
            model = model_class(num_classes=NUM_CLASSES, pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print(f"✓ Model loaded from {model_path}")
            return model, True
        else:
            print(f"⚠️  Checkpoint not found, creating new model")
            model = model_class(num_classes=NUM_CLASSES, pretrained=fallback_pretrained)
            model.to(device)
            return model, False

# Usage
model, loaded = safe_load_model(
    'tumornet_lite_best2.pth',
    TumorNetLite,
    device,
    fallback_pretrained=True
)
```

---

### Example 2: Safe Data Loading with Validation

```python
def safe_load_data(data_path, expected_format='npy'):
    """Load data with validation."""
    with error_handler(f"Loading data from {data_path}", raise_error=False):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if expected_format == 'npy':
            data = np.load(data_path)
        elif expected_format == 'pt':
            data = torch.load(data_path)
        else:
            raise ValueError(f"Unsupported format: {expected_format}")
        
        # Validate
        if data is None or len(data) == 0:
            raise DataValidationError("Loaded data is empty")
        
        print(f"✓ Data loaded: shape {data.shape}")
        return data, True
    
    return None, False

# Usage
x_train, success = safe_load_data('x_train.npy', expected_format='npy')
if not success:
    print("Falling back to data generation...")
    x_train = generate_data()
```

---

### Example 3: Safe Model Inference

```python
def safe_inference(model, data_loader, device, validate_output=True):
    """Perform inference with validation."""
    try:
        model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                # Safe GPU operation
                outputs = safe_gpu_operation(
                    lambda: model(images.to(device)),
                    fallback_device='cpu'
                )
                
                # Validate output
                if validate_output:
                    validate_model_output(
                        outputs,
                        (images.size(0), NUM_CLASSES),
                        f"Batch {batch_idx+1}"
                    )
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        print(f"✓ Inference: {len(all_preds)} samples")
        return all_preds, all_probs, True
        
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        return [], [], False

# Usage
preds, probs, success = safe_inference(model, test_loader, device)
if success:
    accuracy = calculate_accuracy(true_labels, preds)
```

---

### Example 4: Safe Training Loop

```python
def safe_train_epoch(model, train_loader, criterion, optimizer, device, epoch_num):
    """Train one epoch with error handling."""
    try:
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            try:
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # Validate output
                validate_model_output(
                    outputs,
                    (images.size(0), NUM_CLASSES),
                    f"Epoch {epoch_num}, Batch {batch_idx}"
                )
                
                loss = criterion(outputs, labels)
                
                # Check for NaN
                if torch.isnan(loss):
                    raise ModelError(f"NaN loss at batch {batch_idx}")
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"⚠️  GPU OOM at batch {batch_idx}, skipping")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        avg_loss = running_loss / max(num_batches, 1)
        print(f"✓ Epoch {epoch_num}: avg_loss = {avg_loss:.4f}")
        return avg_loss, True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return float('inf'), False

# Usage
for epoch in range(num_epochs):
    avg_loss, success = safe_train_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    
    if not success:
        print(f"Stopping training at epoch {epoch}")
        break
```

---

### Example 5: Safe Metric Calculation

```python
def safe_calculate_metrics(y_true, y_pred):
    """Calculate metrics with validation."""
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Validate inputs
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) != len(y_pred):
            raise DataValidationError(
                f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
            )
        
        # Calculate
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }
        
        # Validate ranges
        for name, value in metrics.items():
            if not (0 <= value <= 1):
                print(f"⚠️  {name} = {value} outside [0,1]")
        
        print("✓ Metrics calculated:")
        for name, value in metrics.items():
            print(f"   {name}: {value:.4f}")
        
        return metrics, True
        
    except Exception as e:
        print(f"❌ Metric calculation failed: {str(e)}")
        return {}, False

# Usage
metrics, success = safe_calculate_metrics(y_true, y_pred)
if success:
    report_results(metrics)
```

---

### Example 6: Safe Visualization

```python
def safe_create_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Create confusion matrix with error handling."""
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Validate
        if len(y_true) != len(y_pred):
            raise DataValidationError("Length mismatch")
        
        if len(y_true) == 0:
            raise DataValidationError("Empty predictions")
        
        # Create
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=14, weight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Safe save
        success = safe_save_figure(save_path, dpi=300)
        plt.close()
        
        return success
        
    except Exception as e:
        print(f"❌ Visualization failed: {str(e)}")
        return False

# Usage
success = safe_create_confusion_matrix(
    y_true, y_pred, class_names,
    'confusion_matrix.png'
)
```

---

## Best Practices

### 1. Always Validate Inputs

```python
# ✅ GOOD: Validate before processing
validate_data_shapes(
    {'images': images, 'labels': labels},
    {'images': (None, 3, 224, 224), 'labels': (None,)},
    operation="batch processing"
)
output = model(images)

# ❌ BAD: Process without validation
output = model(images)  # Could fail with cryptic error
```

---

### 2. Use Error Handler for I/O Operations

```python
# ✅ GOOD: Wrapped with error handler
with error_handler("Loading checkpoint", raise_error=False):
    checkpoint = torch.load('model.pth')

# ❌ BAD: Bare I/O operation
checkpoint = torch.load('model.pth')  # Will crash if file missing
```

---

### 3. Validate Model Outputs

```python
# ✅ GOOD: Check for NaN/Inf
output = model(input)
validate_model_output(output, (batch_size, num_classes))

# ❌ BAD: Assume output is valid
output = model(input)
loss = criterion(output, labels)  # Could be NaN
```

---

### 4. Use Safe GPU Operations

```python
# ✅ GOOD: Automatic CPU fallback
result = safe_gpu_operation(
    lambda: heavy_computation(data),
    fallback_device='cpu'
)

# ❌ BAD: Assume GPU always works
result = heavy_computation(data.to('cuda'))  # Could OOM
```

---

### 5. Graceful Degradation

```python
# ✅ GOOD: Provide alternatives
with error_handler("Loading cached results", raise_error=False):
    results = load_cache('results.pkl')

if results is None:
    print("Cache miss - computing from scratch")
    results = compute_results()

# ❌ BAD: Crash on cache miss
results = load_cache('results.pkl')  # Crashes if missing
```

---

### 6. Clear Error Messages

```python
# ✅ GOOD: Descriptive error
if len(predictions) == 0:
    raise DataValidationError(
        "No predictions generated. Check that:\n"
        "  1. Model is in eval() mode\n"
        "  2. Data loader is not empty\n"
        "  3. Model outputs are valid"
    )

# ❌ BAD: Generic error
if len(predictions) == 0:
    raise ValueError("Empty")  # Not helpful
```

---

## Testing

### Test 1: Error Handler

```python
print("Test 1: Error handler with fallback")
print("-" * 50)

# Should print error but continue
with error_handler("Test operation", raise_error=False, fallback_value=42):
    x = 1 / 0  # Will cause error

print(f"Continued execution with fallback")
```

**Expected Output**:
```
Test 1: Error handler with fallback
--------------------------------------------------
❌ ERROR in 'Test operation':
   Type: ZeroDivisionError
   Message: division by zero
   
📍 Traceback:
   [traceback details]
   
⚠️  Continuing with fallback value: 42
Continued execution with fallback
```

---

### Test 2: Data Validation

```python
print("\nTest 2: Data shape validation")
print("-" * 50)

try:
    # Should raise error due to shape mismatch
    validate_data_shapes(
        {'data': np.zeros((100, 50))},
        {'data': (100, 100)},  # Wrong shape
        operation="test validation"
    )
except DataValidationError as e:
    print(f"✓ Caught validation error as expected")
    print(f"   Error: {e}")
```

**Expected Output**:
```
Test 2: Data shape validation
--------------------------------------------------
✓ Caught validation error as expected
   Error: Shape mismatch for 'data' dimension 1 in test validation:
          Expected: (100, 100)
          Actual: (100, 50)
```

---

### Test 3: GPU Fallback

```python
print("\nTest 3: GPU operation with CPU fallback")
print("-" * 50)

# Should work on either GPU or CPU
tensor = torch.randn(100, 100)
result = safe_gpu_operation(
    lambda: tensor.to('cuda') @ tensor.to('cuda'),
    fallback_device='cpu'
)

print(f"✓ Operation completed on device: {result.device}")
print(f"   Result shape: {result.shape}")
```

**Expected Output** (if GPU available):
```
Test 3: GPU operation with CPU fallback
--------------------------------------------------
✓ Operation completed on device: cuda:0
   Result shape: torch.Size([100, 100])
```

**Expected Output** (if GPU not available or OOM):
```
Test 3: GPU operation with CPU fallback
--------------------------------------------------
⚠️  GPU operation failed: CUDA out of memory
   Falling back to cpu...
✓ Operation completed on device: cpu
   Result shape: torch.Size([100, 100])
```

---

### Test 4: Model Output Validation

```python
print("\nTest 4: Model output validation")
print("-" * 50)

# Test valid output
valid_output = torch.randn(32, 4)  # batch=32, classes=4
try:
    validate_model_output(valid_output, (32, 4), "test inference")
    print("✓ Valid output passed validation")
except ModelError as e:
    print(f"❌ Unexpected error: {e}")

# Test invalid output (contains NaN)
invalid_output = torch.tensor([[1.0, float('nan'), 2.0, 3.0]])
try:
    validate_model_output(invalid_output, (1, 4), "test inference")
    print("❌ Invalid output should have failed")
except ModelError as e:
    print(f"✓ Caught invalid output: {e}")
```

**Expected Output**:
```
Test 4: Model output validation
--------------------------------------------------
✓ Valid output passed validation
✓ Caught invalid output: test inference: Output contains NaN values
```

---

### Test 5: Safe Figure Saving

```python
print("\nTest 5: Safe figure saving")
print("-" * 50)

# Create simple plot
plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Test Plot')

# Test successful save
success = safe_save_figure('test_plot.png', dpi=300)
if success:
    print("✓ Figure saved successfully")
    
# Test failed save (invalid path)
success = safe_save_figure('/invalid/path/plot.png', dpi=300)
if not success:
    print("✓ Failed save handled gracefully")
```

**Expected Output**:
```
Test 5: Safe figure saving
--------------------------------------------------
✓ Figure saved: test_plot.png
✓ Figure saved successfully
❌ Failed to save figure '/invalid/path/plot.png': [Errno 2] No such file or directory
✓ Failed save handled gracefully
```

---

## Summary

### Key Improvements

1. **Robustness**: Notebook won't crash on minor errors
2. **Debugging**: Clear error messages with full context
3. **Flexibility**: Automatic fallbacks (GPU→CPU, cache→compute)
4. **Validation**: Catch problems early with shape/type/value checks
5. **Production-Ready**: Professional error handling standards

### When to Use Each Utility

| Utility | Use Case |
|---------|----------|
| `error_handler()` | Any operation that might fail (I/O, computation) |
| `validate_data_shapes()` | Before processing arrays/tensors |
| `validate_model_output()` | After model forward pass |
| `safe_gpu_operation()` | GPU-intensive operations |
| `safe_save_figure()` | Saving matplotlib plots |
| `check_dependencies()` | At notebook start |
| `memory_check()` | Before large operations |

### Application Checklist

- [ ] Add `check_dependencies()` at notebook start
- [ ] Wrap all file I/O with `error_handler()`
- [ ] Validate data shapes after loading
- [ ] Validate model outputs in training loop
- [ ] Use `safe_gpu_operation()` for GPU code
- [ ] Use `safe_save_figure()` for all plots
- [ ] Add `memory_check()` before large operations
- [ ] Test error paths (missing files, wrong shapes, etc.)

---

**Your notebook is now production-ready with comprehensive error handling!** 🛡️✨
