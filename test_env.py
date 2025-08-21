"""Simple vLLM inference test script"""

import time
from vllm import LLM, SamplingParams
import torch

def test_vllm_basic():
    """Test basic vLLM functionality"""
    print("🚀 Testing vLLM Basic Inference")
    print("=" * 50)
    
    try:
        # Initialize model
        print("📦 Loading model...")
        start_time = time.time()
        
        llm = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            max_model_len=2048,
            enforce_eager=True,  # Disable CUDA graphs for debugging
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=200,
            top_p=0.9,
            stop_token_ids=[llm.get_tokenizer().eos_token_id],
        )
        
        # Test prompts
        test_prompts = [
            "What is artificial intelligence?",
            "Explain how solar panels work.",
            "Write a short poem about coding."
        ]
        
        print(f"\n🔍 Testing with {len(test_prompts)} prompts...")
        
        # Generate responses
        start_time = time.time()
        outputs = llm.generate(test_prompts, sampling_params)
        generation_time = time.time() - start_time
        
        print(f"✅ Generation completed in {generation_time:.2f} seconds")
        print(f"📊 Speed: {len(test_prompts)/generation_time:.2f} prompts/second")
        
        # Display results
        print("\n📝 Generated Responses:")
        print("=" * 50)
        
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            
            print(f"\n🔹 Prompt {i+1}: {prompt}")
            print(f"🔸 Response: {generated_text[:150]}{'...' if len(generated_text) > 150 else ''}")
            print(f"📏 Length: {len(generated_text)} characters")
        
        print("\n✅ vLLM basic test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during basic test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vllm_chat_template():
    """Test vLLM with chat template"""
    print("\n🚀 Testing vLLM with Chat Template")
    print("=" * 50)
    
    try:
        # Initialize model
        print("📦 Loading model with chat template...")
        
        llm = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            max_model_len=2048,
            enforce_eager=True,
        )
        
        # Get tokenizer for chat template
        tokenizer = llm.get_tokenizer()
        
        # Create chat format prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ]
        
        # Apply chat template
        chat_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        print(f"🔍 Chat template applied:")
        print(f"📝 Formatted prompt: {chat_prompt[:200]}...")
        
        # Generate response
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=150,
            top_p=0.9,
        )
        
        outputs = llm.generate([chat_prompt], sampling_params)
        response = outputs[0].outputs[0].text
        
        print(f"\n✅ Chat response generated:")
        print(f"📝 Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during chat template test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vllm_batch_inference():
    """Test vLLM batch inference performance"""
    print("\n🚀 Testing vLLM Batch Inference")
    print("=" * 50)
    
    try:
        # Initialize model
        llm = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            max_model_len=1024,
            enforce_eager=True,
        )
        
        # Create batch of prompts
        batch_prompts = [
            "What is machine learning?",
            "How does photosynthesis work?",
            "Explain the theory of relativity.",
            "What are the benefits of renewable energy?",
            "How do neural networks learn?",
        ]
        
        sampling_params = SamplingParams(
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
        )
        
        print(f"📊 Testing batch inference with {len(batch_prompts)} prompts...")
        
        # Batch inference
        start_time = time.time()
        outputs = llm.generate(batch_prompts, sampling_params)
        batch_time = time.time() - start_time
        
        print(f"✅ Batch inference completed in {batch_time:.2f} seconds")
        print(f"📈 Throughput: {len(batch_prompts)/batch_time:.2f} prompts/second")
        
        # Calculate token statistics
        total_input_tokens = sum(len(llm.get_tokenizer().encode(prompt)) for prompt in batch_prompts)
        total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        
        print(f"📊 Token Statistics:")
        print(f"   Input tokens: {total_input_tokens}")
        print(f"   Output tokens: {total_output_tokens}")
        print(f"   Total tokens: {total_input_tokens + total_output_tokens}")
        print(f"   Tokens/second: {(total_input_tokens + total_output_tokens)/batch_time:.2f}")
        
        # Show sample outputs
        print(f"\n📝 Sample Outputs:")
        for i, output in enumerate(outputs[:2]):  # Show first 2
            print(f"   {i+1}. {output.outputs[0].text[:80]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during batch test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_info():
    """Display GPU information"""
    print("\n🖥️  GPU Information")
    print("=" * 30)
    
    try:
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.is_available()}")
            print(f"🔢 GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Memory usage
            print(f"💾 Current GPU memory:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   GPU {i}: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
        else:
            print("❌ CUDA not available")
            
    except Exception as e:
        print(f"❌ Error getting GPU info: {e}")

def main():
    """Run all vLLM tests"""
    print("🧪 vLLM Inference Test Suite")
    print("=" * 60)
    
    # Display system info
    test_gpu_info()
    
    # Run tests
    tests = [
        ("Basic Inference", test_vllm_basic),
        ("Chat Template", test_vllm_chat_template),
        ("Batch Inference", test_vllm_batch_inference),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🔬 Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! vLLM is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()