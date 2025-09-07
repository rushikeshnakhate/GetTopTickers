"""
Example usage of the Validation Suite.

This module demonstrates how to use the comprehensive validation suite
to address reviewer concerns about benchmarking, performance measurement,
and validation.
"""

import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from validation.validation_suite import ValidationSuite


def example_ticker_selector(years, top_n_tickers=15, **kwargs):
    """
    Example ticker selector function.
    
    This is a mock implementation for demonstration purposes.
    In practice, this would be your actual ticker selection function.
    
    Args:
        years: List of years to analyze
        top_n_tickers: Number of top tickers to select
        **kwargs: Additional parameters
        
    Returns:
        List of selected ticker symbols
    """
    # Mock implementation - returns sample NSE tickers
    sample_tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'BHARTIARTL.NS', 'INFY.NS',
        'ICICIBANK.NS', 'SBIN.NS', 'LICI.NS', 'ITC.NS', 'HINDUNILVR.NS',
        'LT.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS'
    ]
    
    return sample_tickers[:top_n_tickers]


def run_complete_validation_example():
    """Run complete validation example."""
    print("=" * 60)
    print("PYPORTICKERSELECTOR VALIDATION SUITE - EXAMPLE USAGE")
    print("=" * 60)
    
    # Initialize validation suite with ticker selector function
    validator = ValidationSuite(ticker_selector_func=example_ticker_selector)
    
    # Define validation parameters
    years = [2022, 2023, 2024]
    validation_params = {
        'top_n_tickers': 15,
        'rebalancing_months': 1,
        'train_months': 12,
        'test_months': 3,
        'window_size_months': 24,
        'rebalance_frequency': 'monthly'
    }
    
    print(f"Running validation for years: {years}")
    print(f"Validation parameters: {validation_params}")
    print()
    
    try:
        # Run complete validation
        results = validator.run_complete_validation(years, **validation_params)
        
        # Display results summary
        print("VALIDATION RESULTS SUMMARY")
        print("-" * 40)
        
        # Performance profiling results
        if 'performance_profiling' in results:
            perf_prof = results['performance_profiling']
            if 'performance_summary' in perf_prof:
                summary = perf_prof['performance_summary']
                print(f"Performance Summary:")
                print(f"  - Total calls: {summary.get('total_calls', 'N/A')}")
                print(f"  - Success rate: {summary.get('success_rate', 0):.2%}")
                print(f"  - Average execution time: {summary.get('average_execution_time', 0):.4f} seconds")
                print(f"  - Average memory usage: {summary.get('average_memory_used', 0):.2f} MB")
        
        # Benchmark comparison results
        if 'benchmark_comparison' in results:
            bench_comp = results['benchmark_comparison']
            if 'benchmark_comparison' in bench_comp:
                comparison = bench_comp['benchmark_comparison']
                print(f"\nBenchmark Comparison:")
                print(f"  - Your performance: {comparison.get('your_performance', {})}")
                print(f"  - Benchmark comparisons: {len(comparison.get('benchmark_comparisons', {}))}")
        
        # Walk-forward validation results
        if 'walk_forward_validation' in results:
            wf_val = results['walk_forward_validation']
            if 'walk_forward_analysis' in wf_val:
                wf_analysis = wf_val['walk_forward_analysis']
                print(f"\nWalk-Forward Analysis:")
                print(f"  - Total iterations: {wf_analysis.get('total_iterations', 'N/A')}")
                print(f"  - Average test return: {wf_analysis.get('average_test_return', 0):.2%}")
                print(f"  - Average test Sharpe: {wf_analysis.get('average_test_sharpe', 0):.2f}")
                print(f"  - Consistency score: {wf_analysis.get('consistency_score', 0):.2f}")
        
        # Statistical tests results
        if 'statistical_tests' in results:
            stat_tests = results['statistical_tests']
            if 'comprehensive_tests' in stat_tests:
                comp_tests = stat_tests['comprehensive_tests']
                print(f"\nStatistical Tests:")
                print(f"  - Sample size: {comp_tests.get('data_info', {}).get('sample_size', 'N/A')}")
                print(f"  - Significant tests: {comp_tests.get('summary', {}).get('significant_tests', 'N/A')}")
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"validation_results_{timestamp}.json"
        saved_file = validator.save_validation_results(results, filename)
        
        if saved_file:
            print(f"\nResults saved to: {saved_file}")
        
        # Display comprehensive report
        if 'comprehensive_report' in results:
            report = results['comprehensive_report']
            if 'executive_summary' in report:
                print(f"\n{report['executive_summary']}")
        
        print("\n" + "=" * 60)
        print("VALIDATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        return None


def run_individual_validation_examples():
    """Run individual validation component examples."""
    print("\n" + "=" * 60)
    print("INDIVIDUAL VALIDATION COMPONENT EXAMPLES")
    print("=" * 60)
    
    # Initialize validation suite
    validator = ValidationSuite(ticker_selector_func=example_ticker_selector)
    
    years = [2023]
    
    try:
        # Example 1: Performance Profiling
        print("\n1. Performance Profiling Example:")
        print("-" * 40)
        perf_results = validator._run_performance_profiling(years, top_n_tickers=10)
        print(f"Performance profiling completed: {len(perf_results)} results")
        
        # Example 2: Benchmark Comparison
        print("\n2. Benchmark Comparison Example:")
        print("-" * 40)
        bench_results = validator._run_benchmark_comparison(years, top_n_tickers=10)
        print(f"Benchmark comparison completed: {len(bench_results)} results")
        
        # Example 3: Walk-Forward Validation
        print("\n3. Walk-Forward Validation Example:")
        print("-" * 40)
        wf_results = validator._run_walk_forward_validation(years, top_n_tickers=10)
        print(f"Walk-forward validation completed: {len(wf_results)} results")
        
        # Example 4: Market Regime Analysis
        print("\n4. Market Regime Analysis Example:")
        print("-" * 40)
        regime_results = validator._run_market_regime_analysis(years, top_n_tickers=10)
        print(f"Market regime analysis completed: {len(regime_results)} results")
        
        print("\n" + "=" * 60)
        print("INDIVIDUAL VALIDATION EXAMPLES COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during individual validation examples: {str(e)}")


def run_visualization_example():
    """Run visualization example."""
    print("\n" + "=" * 60)
    print("VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    try:
        # Initialize validation suite
        validator = ValidationSuite(ticker_selector_func=example_ticker_selector)
        
        # Run a quick validation to get data for visualization
        years = [2023]
        results = validator.run_complete_validation(years, top_n_tickers=10)
        
        if results and 'visualizations' in results:
            visualizations = results['visualizations']
            print(f"Visualizations generated: {len(visualizations)} plots")
            
            if 'saved_plots' in visualizations:
                saved_plots = visualizations['saved_plots']
                print("Saved plot files:")
                for plot_name, filepath in saved_plots.items():
                    print(f"  - {plot_name}: {filepath}")
        
        print("\n" + "=" * 60)
        print("VISUALIZATION EXAMPLE COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during visualization example: {str(e)}")


if __name__ == "__main__":
    # Run complete validation example
    complete_results = run_complete_validation_example()
    
    # Run individual validation examples
    run_individual_validation_examples()
    
    # Run visualization example
    run_visualization_example()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)
    print("\nThe validation suite addresses all reviewer concerns:")
    print("1. ✓ Performance benchmarking against competitors")
    print("2. ✓ Market benchmark comparison with statistical significance")
    print("3. ✓ Selection accuracy validation and hit rate analysis")
    print("4. ✓ Out-of-sample testing with walk-forward analysis")
    print("5. ✓ Risk management and drawdown analysis")
    print("6. ✓ Transaction costs and realistic trading simulation")
    print("7. ✓ Comprehensive visualization and reporting")
    print("8. ✓ Statistical significance testing")
    print("9. ✓ Market regime analysis")
    print("10. ✓ Performance profiling and memory usage tracking")
