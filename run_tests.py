#!/usr/bin/env python3
"""
Test runner script for the MCP-AutoGen DocQA application.

This script provides convenient commands to run different types of tests
with appropriate configurations and coverage reporting.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} completed successfully")
        return True


def main():
    """Main test runner function."""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py <command>")
        print("\nAvailable commands:")
        print("  unit          - Run unit tests only")
        print("  integration   - Run integration tests only")
        print("  performance   - Run performance tests only")
        print("  all           - Run all tests")
        print("  coverage      - Run all tests with coverage report")
        print("  lint          - Run linting checks")
        print("  format        - Format code with black")
        print("  type-check    - Run type checking with mypy")
        print("  clean         - Clean up test artifacts")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    project_root = Path(__file__).parent
    
    # Ensure we're in the project root
    os.chdir(project_root)
    
    if command == "unit":
        success = run_command(
            ["python", "-m", "pytest", "test/unit/", "-v", "-m", "unit"],
            "Unit Tests"
        )
    
    elif command == "integration":
        success = run_command(
            ["python", "-m", "pytest", "test/integration/", "-v", "-m", "integration"],
            "Integration Tests"
        )
    
    elif command == "performance":
        success = run_command(
            ["python", "-m", "pytest", "test/performance/", "-v", "-m", "performance", "--run-performance"],
            "Performance Tests"
        )
    
    elif command == "all":
        success = run_command(
            ["python", "-m", "pytest", "test/", "-v"],
            "All Tests"
        )
    
    elif command == "coverage":
        success = run_command(
            ["python", "-m", "pytest", "test/", "--cov=backend", "--cov=agents", 
             "--cov=mcp_server", "--cov=storage", "--cov=shared", 
             "--cov-report=html", "--cov-report=term-missing", "-v"],
            "All Tests with Coverage"
        )
        if success:
            print("\n📊 Coverage report generated in htmlcov/index.html")
    
    elif command == "lint":
        print("Running linting checks...")
        success = True
        
        # Run flake8
        success &= run_command(
            ["python", "-m", "flake8", "backend/", "agents/", "mcp_server/", "storage/", "shared/", "test/"],
            "Flake8 Linting"
        )
        
        # Run isort
        success &= run_command(
            ["python", "-m", "isort", "--check-only", "backend/", "agents/", "mcp_server/", "storage/", "shared/", "test/"],
            "Import Sorting Check"
        )
    
    elif command == "format":
        print("Formatting code...")
        success = True
        
        # Run black
        success &= run_command(
            ["python", "-m", "black", "backend/", "agents/", "mcp_server/", "storage/", "shared/", "test/"],
            "Code Formatting with Black"
        )
        
        # Run isort
        success &= run_command(
            ["python", "-m", "isort", "backend/", "agents/", "mcp_server/", "storage/", "shared/", "test/"],
            "Import Sorting with isort"
        )
    
    elif command == "type-check":
        success = run_command(
            ["python", "-m", "mypy", "backend/", "agents/", "mcp_server/", "storage/", "shared/"],
            "Type Checking with MyPy"
        )
    
    elif command == "clean":
        print("Cleaning up test artifacts...")
        import shutil
        
        # Remove test artifacts
        artifacts = [
            "htmlcov/",
            ".pytest_cache/",
            ".coverage",
            "test_document.txt",
            "test.html",
            "debug_*.py"
        ]
        
        for artifact in artifacts:
            if os.path.exists(artifact):
                if os.path.isdir(artifact):
                    shutil.rmtree(artifact)
                    print(f"Removed directory: {artifact}")
                else:
                    os.remove(artifact)
                    print(f"Removed file: {artifact}")
        
        print("✅ Cleanup completed")
        success = True
    
    else:
        print(f"Unknown command: {command}")
        success = False
    
    if success:
        print(f"\n🎉 {command.title()} completed successfully!")
        sys.exit(0)
    else:
        print(f"\n💥 {command.title()} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
