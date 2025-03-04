#!/usr/bin/env python
"""
Natural Gas Trading System - System Health Check

This script performs a comprehensive health check of the natural gas trading system,
verifying API connections, data availability, and component functionality.
"""

import sys
import os
import yaml
import requests
import pandas as pd
import importlib
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import components
from src.utils.logger import get_logger

# Configure logging
logger = get_logger("system_health_check")


class SystemHealthCheck:
    """Class to perform health checks on the trading system."""
    
    def __init__(self):
        """Initialize the health check system."""
        self.project_root = Path(__file__).resolve().parent
        self.results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": "UNKNOWN",
            "checks": {}
        }
        
        # Load credentials if available
        self.credentials = self._load_credentials()
    
    def _load_credentials(self):
        """Load API credentials from config file."""
        credentials_path = self.project_root / "config" / "credentials.yaml"
        if not credentials_path.exists():
            logger.warning(f"Credentials file not found at {credentials_path}")
            return {}
        
        try:
            with open(credentials_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            return {}
    
    def check_directory_structure(self):
        """Check if the project directory structure is correct."""
        logger.info("Checking directory structure...")
        
        required_dirs = [
            "src/data_ingestion",
            "src/data_processing",
            "src/strategies",
            "src/backtesting",
            "src/live_trading",
            "src/utils",
            "data/raw",
            "data/processed",
            "config",
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        status = "PASS" if not missing_dirs else "FAIL"
        self.results["checks"]["directory_structure"] = {
            "status": status,
            "details": {
                "missing_directories": missing_dirs
            }
        }
        
        logger.info(f"Directory structure check: {status}")
        return status == "PASS"
    
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        logger.info("Checking dependencies...")
        
        required_packages = [
            "numpy", "pandas", "matplotlib", "requests", "pyyaml",
            "scikit-learn", "schedule", "plotly"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        status = "PASS" if not missing_packages else "FAIL"
        self.results["checks"]["dependencies"] = {
            "status": status,
            "details": {
                "missing_packages": missing_packages
            }
        }
        
        logger.info(f"Dependencies check: {status}")
        return status == "PASS"
    
    def check_api_connections(self):
        """Check if API connections are working."""
        logger.info("Checking API connections...")
        
        api_statuses = {}
        
        # Check NOAA API
        if "noaa_token" in self.credentials:
            noaa_status = self._check_noaa_api()
            api_statuses["noaa"] = noaa_status
        else:
            api_statuses["noaa"] = {
                "status": "SKIP",
                "message": "NOAA token not found in credentials"
            }
        
        # Check EIA API
        if "eia_api_key" in self.credentials:
            eia_status = self._check_eia_api()
            api_statuses["eia"] = eia_status
        else:
            api_statuses["eia"] = {
                "status": "SKIP",
                "message": "EIA API key not found in credentials"
            }
        
        # Determine overall API status
        api_check_status = "PASS"
        for api, status in api_statuses.items():
            if status["status"] == "FAIL":
                api_check_status = "FAIL"
                break
        
        self.results["checks"]["api_connections"] = {
            "status": api_check_status,
            "details": api_statuses
        }
        
        logger.info(f"API connections check: {api_check_status}")
        return api_check_status == "PASS"
    
    def _check_noaa_api(self):
        """Check NOAA API connection."""
        try:
            token = self.credentials.get("noaa_token")
            headers = {"token": token}
            url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets"
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return {
                    "status": "PASS",
                    "message": "NOAA API connection successful"
                }
            else:
                return {
                    "status": "FAIL",
                    "message": f"NOAA API returned status code {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Error connecting to NOAA API: {str(e)}"
            }
    
    def _check_eia_api(self):
        """Check EIA API connection."""
        try:
            api_key = self.credentials.get("eia_api_key")
            url = f"https://api.eia.gov/v2/natural-gas/pri/sum/data/?api_key={api_key}&frequency=monthly&data[0]=value&facets[duoarea][]=NUS&start=2022-01&end=2022-02&sort[0][column]=period&sort[0][direction]=desc"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return {
                    "status": "PASS",
                    "message": "EIA API connection successful"
                }
            else:
                return {
                    "status": "FAIL",
                    "message": f"EIA API returned status code {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Error connecting to EIA API: {str(e)}"
            }
    
    def check_data_availability(self):
        """Check if required data files are available."""
        logger.info("Checking data availability...")
        
        data_files = {
            "eia_prices": self.project_root / "data" / "raw" / "eia" / "ng_prices.csv",
            "eia_storage": self.project_root / "data" / "raw" / "eia" / "ng_storage.csv",
            "noaa_weather": self.project_root / "data" / "raw" / "noaa" / "weather_data.csv"
        }
        
        missing_files = []
        for name, file_path in data_files.items():
            if not file_path.exists():
                missing_files.append(name)
        
        # Check if we have synthetic data as fallback
        has_synthetic = (self.project_root / "data" / "synthetic").exists()
        
        status = "PASS"
        if missing_files:
            if has_synthetic:
                status = "WARNING"
                message = "Some data files missing, but synthetic data is available"
            else:
                status = "FAIL"
                message = "Data files missing and no synthetic data available"
        else:
            message = "All data files available"
        
        self.results["checks"]["data_availability"] = {
            "status": status,
            "details": {
                "message": message,
                "missing_files": missing_files,
                "has_synthetic_data": has_synthetic
            }
        }
        
        logger.info(f"Data availability check: {status}")
        return status != "FAIL"
    
    def check_component_imports(self):
        """Check if all system components can be imported."""
        logger.info("Checking component imports...")
        
        components = [
            "src.data_ingestion.example_api_usage",
            "src.data_processing.data_pipeline",
            "src.data_processing.synthetic_data",
            "src.strategies.alpha_model",
            "src.strategies.risk_model",
            "src.strategies.weather_storage_strategy",
            "src.backtesting.backtester",
            "src.live_trading.signal_generator",
            "src.utils.logger"
        ]
        
        failed_imports = []
        for component in components:
            try:
                importlib.import_module(component)
            except ImportError as e:
                failed_imports.append({
                    "component": component,
                    "error": str(e)
                })
        
        status = "PASS" if not failed_imports else "FAIL"
        self.results["checks"]["component_imports"] = {
            "status": status,
            "details": {
                "failed_imports": failed_imports
            }
        }
        
        logger.info(f"Component imports check: {status}")
        return status == "PASS"
    
    def run_all_checks(self):
        """Run all health checks and determine overall system status."""
        logger.info("Running all system health checks...")
        
        checks = [
            self.check_directory_structure,
            self.check_dependencies,
            self.check_api_connections,
            self.check_data_availability,
            self.check_component_imports
        ]
        
        results = []
        for check in checks:
            results.append(check())
        
        # Determine overall status
        if all(results):
            self.results["overall_status"] = "PASS"
        elif any(results):
            self.results["overall_status"] = "WARNING"
        else:
            self.results["overall_status"] = "FAIL"
        
        logger.info(f"Overall system health: {self.results['overall_status']}")
        return self.results
    
    def print_report(self):
        """Print a formatted health check report."""
        print("\n" + "="*80)
        print(f"NATURAL GAS TRADING SYSTEM - HEALTH CHECK REPORT")
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Overall Status: {self.results['overall_status']}")
        print("="*80)
        
        for check_name, check_result in self.results["checks"].items():
            status = check_result["status"]
            status_color = {
                "PASS": "\033[92m",  # Green
                "WARNING": "\033[93m",  # Yellow
                "FAIL": "\033[91m",  # Red
                "SKIP": "\033[94m"   # Blue
            }.get(status, "")
            reset_color = "\033[0m"
            
            print(f"\n{check_name.replace('_', ' ').upper()}: {status_color}{status}{reset_color}")
            
            if "details" in check_result:
                for key, value in check_result["details"].items():
                    if isinstance(value, list):
                        if value:
                            print(f"  {key}:")
                            for item in value:
                                if isinstance(item, dict):
                                    for k, v in item.items():
                                        print(f"    - {k}: {v}")
                                else:
                                    print(f"    - {item}")
                        else:
                            print(f"  {key}: None")
                    else:
                        print(f"  {key}: {value}")
        
        print("\n" + "="*80)
    
    def save_report(self, output_path=None):
        """Save the health check report to a file."""
        if output_path is None:
            output_path = self.project_root / "system_health_report.yaml"
        
        try:
            with open(output_path, 'w') as file:
                yaml.dump(self.results, file, default_flow_style=False)
            logger.info(f"Health check report saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving health check report: {e}")
            return False


def main():
    """Main entry point for the health check script."""
    health_check = SystemHealthCheck()
    health_check.run_all_checks()
    health_check.print_report()
    health_check.save_report()


if __name__ == "__main__":
    main() 