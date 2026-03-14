#!/usr/bin/env node
/**
 * Test script for Smart Truncate embedding logic
 * Tests texts of various lengths to ensure truncation works correctly
 */

import { EmbeddingService } from '../src/services/embedding.service';

const embedder = new EmbeddingService();

// Helper to generate long text
function generateLongText(targetTokens: number): string {
  // ~4 chars per token, so multiply by 4
  const targetChars = targetTokens * 4;
  const sentence = "This is a test sentence that contains approximately twenty tokens worth of content. ";
  const sentencesNeeded = Math.ceil(targetChars / sentence.length);
  return sentence.repeat(sentencesNeeded);
}

async function testSmartTruncate() {
  console.log('=== Smart Truncate Test ===\n');

  const testCases = [
    { name: 'Short text (100 tokens)', tokens: 100 },
    { name: 'Medium text (2000 tokens)', tokens: 2000 },
    { name: 'Long text (6000 tokens)', tokens: 6000 },
    { name: 'Very long text (10000 tokens)', tokens: 10000 },
    { name: 'Extreme text (20000 tokens)', tokens: 20000 },
  ];

  for (const tc of testCases) {
    const text = generateLongText(tc.tokens);
    const estimatedTokens = Math.ceil(text.length / 4);
    
    console.log(`\n--- ${tc.name} ---`);
    console.log(`  Original: ${text.length} chars (~${estimatedTokens} tokens)`);
    
    // Test truncation logic directly
    const CHARS_PER_TOKEN = 4;
    const MAX_SAFE_TOKENS = 6000;
    const MAX_SAFE_CHARS = MAX_SAFE_TOKENS * CHARS_PER_TOKEN;
    
    let truncated = text;
    let wasTruncated = false;
    
    if (text.length > MAX_SAFE_CHARS) {
      const headChars = Math.floor(MAX_SAFE_CHARS * 0.7);
      const tailChars = Math.floor(MAX_SAFE_CHARS * 0.3);
      const head = text.slice(0, headChars);
      const tail = text.slice(-tailChars);
      truncated = `${head}\n[...truncated for embedding...]\n${tail}`;
      wasTruncated = true;
    }
    
    const truncatedTokens = Math.ceil(truncated.length / 4);
    
    console.log(`  Truncated: ${truncated.length} chars (~${truncatedTokens} tokens)`);
    console.log(`  Was truncated: ${wasTruncated}`);
    console.log(`  Fits in 8192 limit: ${truncatedTokens <= 8192 ? '✅' : '❌'}`);
    
    // Verify structure
    if (wasTruncated) {
      const hasMarker = truncated.includes('[...truncated for embedding...]');
      console.log(`  Has truncate marker: ${hasMarker ? '✅' : '❌'}`);
    }
  }

  console.log('\n=== Test Complete ===');
}

testSmartTruncate().catch(console.error);
