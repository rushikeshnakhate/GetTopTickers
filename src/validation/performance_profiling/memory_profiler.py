"""
Memory Profiler for detailed memory usage tracking.

This module provides specialized memory profiling capabilities for tracking
memory usage patterns and identifying memory leaks.
"""

import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
import logging
import threading
import time
from contextlib import contextmanager
import gc

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Detailed memory usage profiling and analysis."""
    
    def __init__(self):
        """Initialize the MemoryProfiler."""
        self.memory_snapshots = []
        self.memory_timeline = []
        self._monitoring = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
    def start_memory_monitoring(self, interval: float = 0.1):
        """
        Start continuous memory monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            logger.warning("Memory monitoring already active")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_memory, 
            args=(interval,), 
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started memory monitoring with {interval}s interval")
    
    def stop_memory_monitoring(self):
        """Stop continuous memory monitoring."""
        if not self._monitoring:
            logger.warning("Memory monitoring not active")
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        logger.info("Stopped memory monitoring")
    
    def take_memory_snapshot(self, label: str = "") -> Dict:
        """
        Take a detailed memory snapshot.
        
        Args:
            label: Label for the snapshot
            
        Returns:
            Dictionary with memory information
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            snapshot = {
                'timestamp': time.time(),
                'label': label,
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': memory_percent,
                'system_total_gb': system_memory.total / 1024 / 1024 / 1024,
                'system_available_gb': system_memory.available / 1024 / 1024 / 1024,
                'system_percent': system_memory.percent
            }
            
            with self._lock:
                self.memory_snapshots.append(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error taking memory snapshot: {str(e)}")
            return {}
    
    def profile_memory_usage(self, func: Callable, *args, **kwargs) -> Dict:
        """
        Profile memory usage of a specific function.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Dictionary with memory usage statistics
        """
        # Force garbage collection
        gc.collect()
        
        # Take initial snapshot
        initial_snapshot = self.take_memory_snapshot("initial")
        
        # Start monitoring
        self.start_memory_monitoring(interval=0.01)
        
        try:
            result = func(*args, **kwargs)
            success = True
            error_msg = None
        except Exception as e:
            result = None
            success = False
            error_msg = str(e)
        finally:
            # Stop monitoring
            self.stop_memory_monitoring()
            
            # Take final snapshot
            final_snapshot = self.take_memory_snapshot("final")
            
            # Force garbage collection again
            gc.collect()
            cleanup_snapshot = self.take_memory_snapshot("cleanup")
        
        # Calculate memory statistics
        memory_stats = self._calculate_memory_stats(
            initial_snapshot, final_snapshot, cleanup_snapshot
        )
        
        memory_stats.update({
            'success': success,
            'error_message': error_msg,
            'function_name': func.__name__
        })
        
        return memory_stats
    
    def get_memory_timeline(self) -> pd.DataFrame:
        """
        Get memory usage timeline as DataFrame.
        
        Returns:
            DataFrame with memory timeline
        """
        if not self.memory_timeline:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.memory_timeline)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    
    def get_memory_snapshots(self) -> pd.DataFrame:
        """
        Get memory snapshots as DataFrame.
        
        Returns:
            DataFrame with memory snapshots
        """
        if not self.memory_snapshots:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.memory_snapshots)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    
    def analyze_memory_patterns(self) -> Dict:
        """
        Analyze memory usage patterns.
        
        Returns:
            Dictionary with memory analysis
        """
        if not self.memory_timeline:
            return {}
        
        df = pd.DataFrame(self.memory_timeline)
        
        analysis = {
            'peak_memory_mb': df['rss_mb'].max(),
            'min_memory_mb': df['rss_mb'].min(),
            'avg_memory_mb': df['rss_mb'].mean(),
            'memory_volatility': df['rss_mb'].std(),
            'memory_growth_rate': self._calculate_growth_rate(df['rss_mb']),
            'memory_leak_indicators': self._detect_memory_leaks(df),
            'memory_spikes': self._detect_memory_spikes(df)
        }
        
        return analysis
    
    def _monitor_memory(self, interval: float):
        """Internal method for continuous memory monitoring."""
        while self._monitoring:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                
                snapshot = {
                    'timestamp': time.time(),
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024,
                    'percent': process.memory_percent()
                }
                
                with self._lock:
                    self.memory_timeline.append(snapshot)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {str(e)}")
                break
    
    def _calculate_memory_stats(self, initial: Dict, final: Dict, cleanup: Dict) -> Dict:
        """
        Calculate memory usage statistics.
        
        Args:
            initial: Initial memory snapshot
            final: Final memory snapshot
            cleanup: Cleanup memory snapshot
            
        Returns:
            Dictionary with memory statistics
        """
        if not all([initial, final, cleanup]):
            return {}
        
        return {
            'initial_memory_mb': initial.get('rss_mb', 0),
            'final_memory_mb': final.get('rss_mb', 0),
            'cleanup_memory_mb': cleanup.get('rss_mb', 0),
            'memory_used_mb': final.get('rss_mb', 0) - initial.get('rss_mb', 0),
            'memory_retained_mb': cleanup.get('rss_mb', 0) - initial.get('rss_mb', 0),
            'memory_freed_mb': final.get('rss_mb', 0) - cleanup.get('rss_mb', 0),
            'peak_memory_mb': max(
                initial.get('rss_mb', 0),
                final.get('rss_mb', 0),
                cleanup.get('rss_mb', 0)
            )
        }
    
    def _calculate_growth_rate(self, memory_series: pd.Series) -> float:
        """
        Calculate memory growth rate.
        
        Args:
            memory_series: Series of memory values
            
        Returns:
            Growth rate per second
        """
        if len(memory_series) < 2:
            return 0.0
        
        # Linear regression to find growth rate
        x = np.arange(len(memory_series))
        y = memory_series.values
        
        # Calculate slope (growth rate)
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _detect_memory_leaks(self, df: pd.DataFrame) -> Dict:
        """
        Detect potential memory leaks.
        
        Args:
            df: DataFrame with memory timeline
            
        Returns:
            Dictionary with leak detection results
        """
        if len(df) < 10:
            return {'leak_detected': False, 'confidence': 0.0}
        
        memory_series = df['rss_mb']
        
        # Check for consistent upward trend
        growth_rate = self._calculate_growth_rate(memory_series)
        
        # Check for increasing variance (memory fragmentation)
        window_size = min(50, len(memory_series) // 4)
        if window_size > 1:
            rolling_std = memory_series.rolling(window=window_size).std()
            variance_trend = np.polyfit(range(len(rolling_std.dropna())), 
                                      rolling_std.dropna().values, 1)[0]
        else:
            variance_trend = 0
        
        # Simple leak detection logic
        leak_detected = growth_rate > 0.1 and variance_trend > 0
        confidence = min(1.0, abs(growth_rate) * 10)
        
        return {
            'leak_detected': leak_detected,
            'confidence': confidence,
            'growth_rate': growth_rate,
            'variance_trend': variance_trend
        }
    
    def _detect_memory_spikes(self, df: pd.DataFrame, threshold: float = 2.0) -> List[Dict]:
        """
        Detect memory spikes.
        
        Args:
            df: DataFrame with memory timeline
            threshold: Spike detection threshold (standard deviations)
            
        Returns:
            List of detected spikes
        """
        if len(df) < 5:
            return []
        
        memory_series = df['rss_mb']
        mean_memory = memory_series.mean()
        std_memory = memory_series.std()
        
        spikes = []
        for idx, value in memory_series.items():
            if abs(value - mean_memory) > threshold * std_memory:
                spikes.append({
                    'index': idx,
                    'timestamp': df.loc[idx, 'timestamp'],
                    'memory_mb': value,
                    'deviation': (value - mean_memory) / std_memory
                })
        
        return spikes
    
    def clear_data(self):
        """Clear all memory profiling data."""
        with self._lock:
            self.memory_snapshots.clear()
            self.memory_timeline.clear()
        
        logger.info("Memory profiling data cleared")
    
    @contextmanager
    def memory_context(self, label: str = ""):
        """
        Context manager for memory profiling.
        
        Args:
            label: Label for the memory profiling context
        """
        initial_snapshot = self.take_memory_snapshot(f"{label}_start")
        
        try:
            yield
        finally:
            final_snapshot = self.take_memory_snapshot(f"{label}_end")
            
            # Log memory usage
            memory_used = final_snapshot.get('rss_mb', 0) - initial_snapshot.get('rss_mb', 0)
            logger.info(f"Memory usage for '{label}': {memory_used:.2f} MB")
