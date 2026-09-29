"""
Data Processing Script for Hydraulic Systems Dataset

This script processes the raw hydraulic systems data and generates:
1. Normalized time series DataFrame (all sensors at 1 Hz)
2. Aggregated statistical features DataFrame
3. Feature metadata JSON for agent consumption

Run this script once to generate processed data files that can be
efficiently loaded by training, agent, and view layers.

Usage:
    python scripts/process_data.py [--verbose] [--force]

Options:
    --verbose: Print detailed progress information
    --force: Force regeneration even if files exist
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from pathlib import Path    
import os
from dotenv import load_dotenv

load_dotenv()


# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from layers.data.src.loaders import load_full_dataset, get_sensor_info
from layers.data.src.preprocessors import (
    normalize_to_base_frequency,
    create_aggregated_features,
    validate_dataframe
)
from layers.data.src.metadata import (
    generate_features_metadata,
    save_metadata,
    create_feature_summary
)


def print_banner(text: str, char: str = "=") -> None:
    """Print a formatted banner."""
    print(f"\n{char * 80}")
    print(f"{text:^80}")
    print(f"{char * 80}\n")


def main():
    """Main processing pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Process hydraulic systems dataset"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if files exist'
    )
    args = parser.parse_args()
    
    verbose = args.verbose
    force = args.force
    
    # Setup paths
    data_layer_path = Path(__file__).parent.parent
    processed_path = data_layer_path / 'processed'
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Define output files
    normalized_long_file = processed_path / 'hydraulic_normalized_long.csv'
    normalized_wide_file = processed_path / 'hydraulic_normalized_wide.csv'
    aggregated_file = processed_path / 'hydraulic_aggregated.csv'
    metadata_file = processed_path / 'features_metadata.json'
    summary_file = processed_path / 'feature_summary.md'
    
    print_banner("Hydraulic Systems Data Processing Pipeline")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Output directory: {processed_path.absolute()}\n")
    
    # Check if files exist
    if not force:
        existing_files = []
        if normalized_long_file.exists():
            existing_files.append('normalized_long')
        if normalized_wide_file.exists():
            existing_files.append('normalized_wide')
        if aggregated_file.exists():
            existing_files.append('aggregated')
        if metadata_file.exists():
            existing_files.append('metadata')
        
        if existing_files:
            print("⚠️  The following files already exist:")
            for fname in existing_files:
                print(f"   - {fname}")
            print("\nUse --force to regenerate")
            
            response = input("\nContinue and skip existing files? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    # Step 1: Load raw data
    print_banner("Step 1: Loading Raw Dataset", "-")
    
    try:
        df_raw = load_full_dataset(verbose=verbose)
        print(f"✅ Raw data loaded successfully!")
        print(f"   Shape: {df_raw.shape}")
        print(f"   Memory: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Validate raw data
        validation = validate_dataframe(df_raw, verbose=verbose)
        if not validation['valid']:
            print("\n⚠️  Validation warnings:")
            for issue in validation['issues']:
                print(f"   - {issue}")
        
    except Exception as e:
        print(f"❌ Error loading raw data: {str(e)}")
        return 1
    
    # Step 2: Create normalized time series
    print_banner("Step 2: Creating Normalized Time Series", "-")
    
    # 2a: Long format (RECOMMENDED - one row per time point)
    if normalized_long_file.exists() and not force:
        print(f"⏭️  Skipping long format (file exists): {normalized_long_file}")
        # Load for next steps
        import pandas as pd
        df_normalized_long = pd.read_csv(normalized_long_file)
    else:
        try:
            print("🔄 Normalizing all sensors to 1 Hz (LONG format)...")
            print("   Each row = one time point (continuous time_sec, sensors, targets)")
            df_normalized_long = normalize_to_base_frequency(
                df_raw,
                base_freq=1,
                aggregation='median',
                format='long',
                verbose=verbose
            )
            
            # Save
            df_normalized_long.to_csv(normalized_long_file, index=False)
            file_size = normalized_long_file.stat().st_size / 1024**2
            
            print(f"✅ Normalized data (long format) saved!")
            print(f"   File: {normalized_long_file.name}")
            print(f"   Shape: {df_normalized_long.shape}")
            print(f"   Expected: (132,300 rows × ~23 cols)")
            print(f"   Size: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"❌ Error creating normalized long format data: {str(e)}")
            return 1
    
    # 2b: Wide format (for compatibility)
    if normalized_wide_file.exists() and not force:
        print(f"⏭️  Skipping wide format (file exists): {normalized_wide_file}")
    else:
        try:
            print("\n🔄 Creating wide format (for compatibility)...")
            print("   Each row = one cycle (sensor_t0000, sensor_t0001, ...)")
            df_normalized_wide = normalize_to_base_frequency(
                df_raw,
                base_freq=1,
                aggregation='median',
                format='wide',
                verbose=verbose
            )
            
            # Save
            df_normalized_wide.to_csv(normalized_wide_file, index=False)
            file_size = normalized_wide_file.stat().st_size / 1024**2
            
            print(f"✅ Normalized data (wide format) saved!")
            print(f"   File: {normalized_wide_file.name}")
            print(f"   Shape: {df_normalized_wide.shape}")
            print(f"   Expected: (2,205 rows × ~1,025 cols)")
            print(f"   Size: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"❌ Error creating normalized wide format data: {str(e)}")
            return 1
    
    # Step 3: Create aggregated features
    print_banner("Step 3: Creating Aggregated Features", "-")
    
    if aggregated_file.exists() and not force:
        print(f"⏭️  Skipping (file exists): {aggregated_file}")
        # Load for metadata generation
        import pandas as pd
        df_aggregated = pd.read_csv(aggregated_file)
    else:
        try:
            print("🔄 Extracting statistical features from time series...")
            df_aggregated = create_aggregated_features(
                df_raw,
                statistics=['mean', 'std', 'min', 'max', 'median', 'range', 'q25', 'q75'],
                verbose=verbose
            )
            
            # Save
            df_aggregated.to_csv(aggregated_file, index=False)
            file_size = aggregated_file.stat().st_size / 1024**2
            
            print(f"✅ Aggregated data saved!")
            print(f"   File: {aggregated_file.name}")
            print(f"   Shape: {df_aggregated.shape}")
            print(f"   Size: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"❌ Error creating aggregated data: {str(e)}")
            return 1
    
    # Step 4: Generate metadata
    print_banner("Step 4: Generating Feature Metadata", "-")
    
    if metadata_file.exists() and not force:
        print(f"⏭️  Skipping (file exists): {metadata_file}")
    else:
        try:
            # Load normalized_long if not already in memory
            if 'df_normalized_long' not in locals():
                import pandas as pd
                df_normalized_long = pd.read_csv(normalized_long_file)
            
            print("🔄 Generating comprehensive feature metadata...")
            metadata = generate_features_metadata(
                df_normalized_long,
                normalized=True,
                domain_file_path= os.getenv('DOMAIN_PATH'),
                include_statistics=True
            )
            
            # Save metadata
            save_metadata(metadata, metadata_file, verbose=verbose)
            file_size = metadata_file.stat().st_size / 1024
            
            print(f"✅ Metadata saved!")
            print(f"   File: {metadata_file.name}")
            print(f"   Features: {len(metadata['features'])}")
            print(f"   Targets: {len(metadata['targets'])}")
            print(f"   Size: {file_size:.2f} KB")
            
            # Create and save summary
            print("\n🔄 Creating feature summary...")
            summary = create_feature_summary(metadata, output_format='markdown')
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            print(f"✅ Summary saved to {summary_file.name}")
            
        except Exception as e:
            print(f"❌ Error generating metadata: {str(e)}")
            return 1
    
    # Final summary
    print_banner("Processing Complete!", "=")
    
    print("📊 Generated Files:")
    
    if normalized_long_file.exists():
        import pandas as pd
        df_temp = pd.read_csv(normalized_long_file)
        print(f"   1. {normalized_long_file.name}")
        print(f"      - Normalized time series (LONG format, 1 Hz)")
        print(f"      - Shape: {df_temp.shape}")
        print(f"      - Use for: Time series visualization, pandas plotting, temporal analysis")
    
    if normalized_wide_file.exists():
        import pandas as pd
        df_temp = pd.read_csv(normalized_wide_file)
        print(f"\n   2. {normalized_wide_file.name}")
        print(f"      - Normalized time series (WIDE format, 1 Hz)")
        print(f"      - Shape: {df_temp.shape}")
        print(f"      - Use for: Compatibility, LSTM (after reshaping)")
    
    if aggregated_file.exists():
        import pandas as pd
        df_temp = pd.read_csv(aggregated_file)
        print(f"\n   3. {aggregated_file.name}")
        print(f"      - Statistical features")
        print(f"      - Shape: {df_temp.shape}")
        print(f"      - Use for: XGBoost, Random Forest, traditional ML")
    
    if metadata_file.exists():
        print(f"\n   4. {metadata_file.name}")
        print(f"      - Feature metadata JSON")
        print(f"      - Use for: Agent layer, documentation")
    
    if summary_file.exists():
        print(f"\n   5. {summary_file.name}")
        print(f"      - Human-readable feature summary")
        print(f"      - Use for: Documentation, reference")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review feature_summary.md for dataset overview")
    print(f"   2. Use get_original_dataframe(variant='normalized', format='long') in training/view layers")
    print(f"   3. Use get_original_dataframe(variant='aggregated') for traditional ML")
    print(f"   4. Use get_features_metadata() in agent layer")
    print(f"   5. Check layers/data/README.md for usage examples")
    
    print(f"\n🕒 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
