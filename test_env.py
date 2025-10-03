import sys

def check_package(name, import_name=None):
    if import_name is None:
        import_name = name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"   ✅ {name}: {version}")
        return True
    except ImportError:
        print(f"   ❌ {name}: MISSING!")
        return False

print("🔍 Complete Environment Verification")
print("=" * 50)

# Critical packages for data collection
print("\n📊 Data Collection Packages:")
data_packages = [
    ('PyTorch', 'torch'),
    ('Transformers', 'transformers'), 
    ('vLLM', 'vllm'),  # CRITICAL!
    ('Datasets', 'datasets'),
    ('NumPy', 'numpy'),
    ('Pandas', 'pandas'),
    ('PyYAML', 'yaml'),
]

data_ok = all(check_package(name, imp) for name, imp in data_packages)

# Critical packages for training
print("\n🎯 Training Packages:")
train_packages = [
    ('TRL', 'trl'),
    ('PEFT', 'peft'),
    ('Accelerate', 'accelerate'),
    ('DeepSpeed', 'deepspeed'),
    ('Wandb', 'wandb'),
]

train_ok = all(check_package(name, imp) for name, imp in train_packages)

# GPU check
print("\n🚀 GPU Setup:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ CUDA: {torch.version.cuda}")
        print(f"   ✅ GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"      GPU {i}: {props.name} ({props.total_memory//1e9:.0f}GB)")
    else:
        print("   ❌ CUDA not available!")
except:
    print("   ❌ Cannot check CUDA")

print(f"\n📋 Summary:")
print(f"   Data Collection Ready: {'✅' if data_ok else '❌'}")
print(f"   Training Ready: {'✅' if train_ok else '❌'}")

if data_ok and train_ok:
    print("\n🎉 Complete pipeline ready!")
    print("   ✅ Data collection with vLLM")
    print("   ✅ Data processing") 
    print("   ✅ GRPO training with TRL")
else:
    print("\n❌ Setup incomplete - install missing packages")
    sys.exit(1)