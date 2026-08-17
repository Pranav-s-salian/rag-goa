"""
Latency benchmark runner: test queries through pipeline and report P50/P70/P100.
"""
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent.parent))

from harness.pipeline import RAGPipeline, PipelineConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LatencyBenchmark:
    """Run latency benchmarks on the RAG pipeline."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize benchmark with pipeline configuration."""
        self.config = config
        self.pipeline = RAGPipeline(config)
        self.results = []
    
    async def run_benchmark(
        self,
        test_queries: List[Dict],
        skip_voice: bool = True
    ) -> Dict:
        """
        Run benchmark on test queries.
        
        Args:
            test_queries: List of test query dicts
            skip_voice: Whether to skip voice queries (requires audio files)
        
        Returns:
            Dictionary with aggregated results
        """
        logger.info(f"Starting latency benchmark with {len(test_queries)} queries")
        
        for i, query_spec in enumerate(test_queries):
            query_id = query_spec['id']
            category = query_spec['category']
            
            # Skip voice queries if requested (require audio files)
            if category == 'voice' and skip_voice:
                logger.info(f"[{i+1}/{len(test_queries)}] Skipping voice query: {query_id}")
                continue
            
            text = query_spec['text']
            if text.startswith('[AUDIO]'):
                text = text.replace('[AUDIO]', '').strip()
            
            logger.info(f"[{i+1}/{len(test_queries)}] Running query: {query_id}")
            
            try:
                # Run pipeline
                result = await self.pipeline.run(query=text)
                
                # Record result
                self.results.append({
                    'query_id': query_id,
                    'category': category,
                    'query': text[:50] + '...',
                    'refused': result.refused,
                    'refusal_reason': result.refusal_reason,
                    'latency': {
                        'stt_ms': result.stt_ms,
                        'retrieval_ms': result.retrieval_ms,
                        'generation_ms': result.generation_ms,
                        'guardrail_ms': result.guardrail_ms,
                        'total_ms': result.total_ms
                    },
                    'error': result.error
                })
                
                logger.info(f"  Result: {'REFUSED' if result.refused else 'ANSWERED'} ({result.total_ms:.2f}ms)")
                
            except Exception as e:
                logger.error(f"  Error: {e}")
                self.results.append({
                    'query_id': query_id,
                    'category': category,
                    'query': text[:50] + '...',
                    'error': str(e)
                })
        
        # Aggregate results
        return self._aggregate_results()
    
    def _aggregate_results(self) -> Dict:
        """Aggregate benchmark results into statistics."""
        logger.info("\nAggregating results...")
        
        # Collect latencies by stage
        stage_latencies = defaultdict(list)
        
        for result in self.results:
            if 'latency' in result:
                latency = result['latency']
                for stage, ms in latency.items():
                    if ms > 0:  # Only include stages that ran
                        stage_latencies[stage].append(ms)
        
        # Calculate percentiles for each stage
        aggregated = {}
        
        for stage, latencies in stage_latencies.items():
            if not latencies:
                continue
            
            aggregated[stage] = {
                'count': len(latencies),
                'mean': float(np.mean(latencies)),
                'std': float(np.std(latencies)),
                'min': float(np.min(latencies)),
                'p50': float(np.percentile(latencies, 50)),
                'p70': float(np.percentile(latencies, 70)),
                'p95': float(np.percentile(latencies, 95)),
                'p100': float(np.max(latencies))
            }
        
        # Category-wise breakdown
        category_stats = defaultdict(lambda: {'total': 0, 'answered': 0, 'refused': 0, 'errors': 0})
        
        for result in self.results:
            category = result['category']
            category_stats[category]['total'] += 1
            
            if result.get('error'):
                category_stats[category]['errors'] += 1
            elif result.get('refused'):
                category_stats[category]['refused'] += 1
            else:
                category_stats[category]['answered'] += 1
        
        return {
            'total_queries': len(self.results),
            'stage_latencies': aggregated,
            'category_breakdown': dict(category_stats),
            'detailed_results': self.results
        }
    
    def save_report(self, results: Dict, output_path: Path):
        """Save benchmark report to files."""
        # Save JSON
        json_path = output_path / "latency_benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved detailed results to {json_path}")
        
        # Generate markdown report
        md_path = output_path / "latency_benchmark_report.md"
        with open(md_path, 'w') as f:
            f.write("# Latency Benchmark Report\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Queries**: {results['total_queries']}\n")
            f.write(f"- **Strategy**: {self.config.strategy_name}\n")
            f.write(f"- **Reranking**: {'Enabled' if self.config.enable_rerank else 'Disabled'}\n\n")
            
            # Stage latencies table
            f.write("## Per-Stage Latency Metrics\n\n")
            f.write("| Stage | Count | Mean (ms) | P50 (ms) | P70 (ms) | P95 (ms) | P100 (ms) | Target |\n")
            f.write("|-------|-------|-----------|----------|----------|----------|-----------|--------|\n")
            
            stage_order = ['stt_ms', 'retrieval_ms', 'generation_ms', 'guardrail_ms', 'total_ms']
            stage_names = {
                'stt_ms': 'STT',
                'retrieval_ms': 'Retrieval',
                'generation_ms': 'Generation',
                'guardrail_ms': 'Guardrails',
                'total_ms': 'Total'
            }
            stage_targets = {
                'retrieval_ms': '< 100ms',
                'total_ms': '< 3000ms'
            }
            
            for stage_key in stage_order:
                if stage_key in results['stage_latencies']:
                    stats = results['stage_latencies'][stage_key]
                    name = stage_names.get(stage_key, stage_key)
                    target = stage_targets.get(stage_key, '-')
                    
                    f.write(
                        f"| {name:12s} | {stats['count']:5d} | "
                        f"{stats['mean']:9.2f} | {stats['p50']:8.2f} | "
                        f"{stats['p70']:8.2f} | {stats['p95']:8.2f} | "
                        f"{stats['p100']:9.2f} | {target} |\n"
                    )
            
            # Category breakdown
            f.write("\n## Query Category Breakdown\n\n")
            f.write("| Category | Total | Answered | Refused | Errors |\n")
            f.write("|----------|-------|----------|---------|--------|\n")
            
            for category, stats in results['category_breakdown'].items():
                f.write(
                    f"| {category:12s} | {stats['total']:5d} | "
                    f"{stats['answered']:8d} | {stats['refused']:7d} | "
                    f"{stats['errors']:6d} |\n"
                )
            
            # Interpretation
            f.write("\n## Interpretation\n\n")
            f.write("### Latency Targets\n\n")
            f.write("- **Retrieval**: P100 < 100ms (ideally < 50ms) ✓/✗\n")
            f.write("- **Total Pipeline**: P95 < 3000ms for acceptable user experience\n\n")
            
            f.write("### Category Expectations\n\n")
            f.write("- **in_domain**: Should answer (from indexed content)\n")
            f.write("- **adversarial**: Should refuse (off-topic or unsafe)\n")
            f.write("- **voice**: Should answer (requires audio files)\n\n")
            
            # Check retrieval target
            if 'retrieval_ms' in results['stage_latencies']:
                retrieval_p100 = results['stage_latencies']['retrieval_ms']['p100']
                if retrieval_p100 < 100:
                    f.write(f"✓ Retrieval target met: P100 = {retrieval_p100:.2f}ms < 100ms\n\n")
                else:
                    f.write(f"✗ Retrieval target NOT met: P100 = {retrieval_p100:.2f}ms >= 100ms\n\n")
        
        logger.info(f"Saved markdown report to {md_path}")


async def main():
    """Run latency benchmark."""
    # Paths
    eval_dir = Path(__file__).parent
    models_dir = eval_dir.parent / "models"
    
    # Load test queries
    test_queries_path = eval_dir / "test_queries.json"
    with open(test_queries_path, 'r') as f:
        test_queries = json.load(f)
    
    logger.info(f"Loaded {len(test_queries)} test queries from {test_queries_path}")
    
    # Configuration
    config = PipelineConfig(
        strategy_name="fixed",  # Change to test different strategies
        models_dir=str(models_dir),
        enable_rerank=False
    )
    
    # Run benchmark
    benchmark = LatencyBenchmark(config)
    results = await benchmark.run_benchmark(test_queries, skip_voice=True)
    
    # Save report
    benchmark.save_report(results, eval_dir)
    
    # Print summary to console
    logger.info("\n" + "="*80)
    logger.info("LATENCY BENCHMARK SUMMARY")
    logger.info("="*80)
    
    if 'stage_latencies' in results:
        for stage, stats in results['stage_latencies'].items():
            logger.info(
                f"{stage:15s}: P50={stats['p50']:7.2f}ms  "
                f"P70={stats['p70']:7.2f}ms  P100={stats['p100']:7.2f}ms"
            )
    
    logger.info("="*80)
    logger.info(f"\nDetailed report saved to {eval_dir}")


if __name__ == "__main__":
    asyncio.run(main())
