#!/usr/bin/env python3
"""
Rule-based reward server for REINFORCE training
"""

import sys
import os
sys.path.append('rl_src')

from reward_utils import compute_binary_reward
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/get_reward', methods=['POST'])
def get_reward():
    """
    Endpoint that computes rule-based binary rewards
    """
    try:
        data = request.json
        
        # Extract generated text and ground truth
        generated_texts = data.get('generated_texts', [])
        ground_truths = data.get('ground_truths', [])
        
        if len(generated_texts) != len(ground_truths):
            return jsonify({'error': 'Mismatched lengths'}), 400
        
        # Compute binary rewards using our rule-based function
        rewards = []
        for gen_text, ground_truth in zip(generated_texts, ground_truths):
            reward = compute_binary_reward(gen_text, ground_truth)
            rewards.append(reward)
        
        return jsonify({'rewards': rewards})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
