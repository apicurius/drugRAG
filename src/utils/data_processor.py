#!/usr/bin/env python3
"""
SIDER Data Processing for RAG-based Drug Side Effect Retrieval
This script processes the SIDER 4.1 database to create the evaluation dataset
exactly as described in the paper.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import random
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SiderDataProcessor:
    def __init__(self, data_dir="sider_data"):
        self.data_dir = Path(data_dir)
        self.drug_names = {}
        self.drug_atc = {}
        self.side_effects = {}
        self.drug_se_associations = defaultdict(set)
        self.all_drugs = set()
        self.all_side_effects = set()
        
    def load_drug_names(self):
        """Load drug names from drug_names.tsv"""
        logging.info("Loading drug names...")
        file_path = self.data_dir / "drug_names.tsv"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cid = parts[0]
                    name = parts[1].lower()  # Convert to lowercase for consistency
                    self.drug_names[cid] = name
        
        logging.info(f"Loaded {len(self.drug_names)} drug names")
        
    def load_drug_atc(self):
        """Load ATC classifications from drug_atc.tsv"""
        logging.info("Loading ATC classifications...")
        file_path = self.data_dir / "drug_atc.tsv"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cid = parts[0]
                    atc = parts[1]
                    if cid not in self.drug_atc:
                        self.drug_atc[cid] = []
                    self.drug_atc[cid].append(atc)
        
        logging.info(f"Loaded ATC classifications for {len(self.drug_atc)} drugs")
        
    def load_side_effects(self):
        """Load side effects from meddra_all_se.tsv"""
        logging.info("Loading side effect associations...")
        file_path = self.data_dir / "meddra_all_se.tsv"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    drug_cid = parts[0]
                    se_type = parts[3]  # LLT or PT
                    se_name = parts[5].lower()  # Convert to lowercase for consistency
                    
                    # Only use Preferred Terms (PT) as mentioned in the paper
                    if se_type == "PT":
                        # Only include drugs that have names and ATC classifications
                        if drug_cid in self.drug_names and drug_cid in self.drug_atc:
                            drug_name = self.drug_names[drug_cid]
                            self.drug_se_associations[drug_name].add(se_name)
                            self.all_drugs.add(drug_name)
                            self.all_side_effects.add(se_name)
        
        logging.info(f"Loaded associations for {len(self.drug_se_associations)} drugs")
        logging.info(f"Total unique drugs: {len(self.all_drugs)}")
        logging.info(f"Total unique side effects: {len(self.all_side_effects)}")
        
    def create_evaluation_dataset(self, min_side_effects=10, pairs_per_drug=20):
        """
        Create balanced evaluation dataset as described in the paper:
        - Focus on drugs with at least 10 known side effect associations
        - For each drug: 10 positive pairs (known associations) + 10 negative pairs (unknown)
        - Target: 19,520 pairs with 976 drugs and 3,851 side effects
        """
        logging.info("Creating evaluation dataset...")
        
        # Filter drugs with at least min_side_effects associations
        qualified_drugs = []
        for drug, side_effects in self.drug_se_associations.items():
            if len(side_effects) >= min_side_effects:
                qualified_drugs.append(drug)
        
        logging.info(f"Found {len(qualified_drugs)} drugs with at least {min_side_effects} side effects")
        
        # Sort for reproducibility
        qualified_drugs.sort()
        
        # Calculate how many drugs we need to get approximately 19,520 pairs
        target_total_pairs = 19520
        target_drugs = target_total_pairs // pairs_per_drug
        
        # Select drugs to reach target (976 drugs mentioned in paper)
        if len(qualified_drugs) > target_drugs:
            # Prioritize drugs with more side effects for better coverage
            drug_se_counts = [(drug, len(self.drug_se_associations[drug])) for drug in qualified_drugs]
            drug_se_counts.sort(key=lambda x: x[1], reverse=True)
            selected_drugs = [drug for drug, _ in drug_se_counts[:target_drugs]]
        else:
            selected_drugs = qualified_drugs
        
        logging.info(f"Selected {len(selected_drugs)} drugs for evaluation")
        
        # Create balanced dataset
        evaluation_data = []
        used_side_effects = set()
        
        for drug in selected_drugs:
            known_side_effects = list(self.drug_se_associations[drug])
            
            # Sample 10 positive associations
            if len(known_side_effects) >= 10:
                positive_ses = random.sample(known_side_effects, 10)
            else:
                # If fewer than 10, use all available
                positive_ses = known_side_effects
                
            # Add positive pairs
            for se in positive_ses:
                evaluation_data.append({
                    'drug': drug,
                    'side_effect': se,
                    'label': 1,  # Positive association
                    'query': f"Is {se} an adverse effect of {drug}?"
                })
                used_side_effects.add(se)
            
            # Sample 10 negative associations (side effects not associated with this drug)
            available_negative_ses = list(self.all_side_effects - set(known_side_effects))
            
            if len(available_negative_ses) >= 10:
                negative_ses = random.sample(available_negative_ses, 10)
            else:
                # If not enough negatives, sample with replacement
                negative_ses = random.choices(available_negative_ses, k=10)
            
            # Add negative pairs
            for se in negative_ses:
                evaluation_data.append({
                    'drug': drug,
                    'side_effect': se,
                    'label': 0,  # Negative association
                    'query': f"Is {se} an adverse effect of {drug}?"
                })
                used_side_effects.add(se)
        
        # Convert to DataFrame
        df = pd.DataFrame(evaluation_data)
        
        # Statistics
        unique_drugs = df['drug'].nunique()
        unique_side_effects = len(used_side_effects)
        total_pairs = len(df)
        positive_pairs = (df['label'] == 1).sum()
        negative_pairs = (df['label'] == 0).sum()
        
        logging.info(f"Evaluation dataset created:")
        logging.info(f"  Total pairs: {total_pairs}")
        logging.info(f"  Unique drugs: {unique_drugs}")
        logging.info(f"  Unique side effects: {unique_side_effects}")
        logging.info(f"  Positive pairs: {positive_pairs}")
        logging.info(f"  Negative pairs: {negative_pairs}")
        
        return df
    
    def create_data_formats(self, evaluation_df):
        """
        Create Data Format A and Data Format B as described in the paper
        """
        logging.info("Creating data formats...")
        
        # Data Format A: Structured, comma-separated list of all known side effects per drug
        format_a_data = []
        for drug in evaluation_df['drug'].unique():
            # Get all known side effects for this drug from original data
            known_ses = list(self.drug_se_associations[drug])
            se_list = ", ".join(sorted(known_ses))
            text_a = f"The drug {drug} causes the following side effects or adverse reactions: {se_list}"
            format_a_data.append({
                'drug': drug,
                'text': text_a,
                'format': 'A'
            })
        
        # Data Format B: Each drug-side effect pair on a new line
        format_b_data = []
        for drug in evaluation_df['drug'].unique():
            known_ses = list(self.drug_se_associations[drug])
            for se in known_ses:
                text_b = f"The drug {drug} may cause {se} as an adverse effect, adverse reaction, or side effect."
                format_b_data.append({
                    'drug': drug,
                    'side_effect': se,
                    'text': text_b,
                    'format': 'B'
                })
        
        format_a_df = pd.DataFrame(format_a_data)
        format_b_df = pd.DataFrame(format_b_data)
        
        logging.info(f"Format A: {len(format_a_df)} entries")
        logging.info(f"Format B: {len(format_b_df)} entries")
        
        return format_a_df, format_b_df
    
    def save_data(self, evaluation_df, format_a_df, format_b_df, output_dir="processed_data"):
        """Save all processed data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save evaluation dataset
        evaluation_df.to_csv(output_path / "evaluation_dataset.csv", index=False)
        
        # Try to save Excel file (optional)
        try:
            evaluation_df.to_excel(output_path / "evaluation_dataset.xlsx", index=False)
        except ImportError:
            logging.warning("openpyxl not available, skipping Excel export")
        
        # Save format A and B
        format_a_df.to_csv(output_path / "data_format_a.csv", index=False)
        format_b_df.to_csv(output_path / "data_format_b.csv", index=False)
        
        # Save as text files for easy reading (matching existing data format)
        with open(output_path / "data_format_a.txt", 'w', encoding='utf-8') as f:
            for _, row in format_a_df.iterrows():
                f.write(f"\n {row['text']}\n")
        
        with open(output_path / "data_format_b.txt", 'w', encoding='utf-8') as f:
            for _, row in format_b_df.iterrows():
                f.write(f"\n {row['text']}\n")
        
        # Save summary statistics
        stats = {
            'total_evaluation_pairs': len(evaluation_df),
            'unique_drugs': evaluation_df['drug'].nunique(),
            'unique_side_effects': evaluation_df['side_effect'].nunique(),
            'positive_pairs': (evaluation_df['label'] == 1).sum(),
            'negative_pairs': (evaluation_df['label'] == 0).sum(),
            'format_a_entries': len(format_a_df),
            'format_b_entries': len(format_b_df)
        }
        
        with open(output_path / "dataset_statistics.txt", 'w') as f:
            f.write("SIDER Dataset Processing Statistics\n")
            f.write("=" * 40 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        logging.info(f"All data saved to {output_path}")
        
        return stats

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize processor
    processor = SiderDataProcessor()
    
    # Load all data
    processor.load_drug_names()
    processor.load_drug_atc()
    processor.load_side_effects()
    
    # Create evaluation dataset
    evaluation_df = processor.create_evaluation_dataset()
    
    # Create data formats
    format_a_df, format_b_df = processor.create_data_formats(evaluation_df)
    
    # Save all data
    stats = processor.save_data(evaluation_df, format_a_df, format_b_df)
    
    print("\nDataset creation completed successfully!")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()