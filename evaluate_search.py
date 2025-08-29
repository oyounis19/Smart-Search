import pandas as pd
import numpy as np
from simple_search import SimpleSmartSearch
from typing import List, Dict
import time

class SearchEvaluator:
    def __init__(self):
        self.search = SimpleSmartSearch()
        
        # Test queries with expected characteristics
        self.test_queries = [
            {
                'query': 'iPhone 13 under 15k with 128GB storage',
                'expected_brand': 'Apple - iPhone',
                'expected_storage': 128,
                'max_price': 15000,
                'description': 'English query with specific constraints'
            },
            {
                'query': 'أيفون استعمال خفيف 256 جيجا',
                'expected_brand': 'Apple - iPhone', 
                'expected_storage': 256,
                'expected_condition': 'good',
                'description': 'Arabic query with storage and condition'
            },
            {
                'query': 'Samsung A52 good battery life',
                'expected_brand': 'Samsung',
                'expected_model': 'a52',
                'description': 'Brand-specific query with quality requirement'
            },
            {
                'query': 'سامسونج نوت جديد بالضمان',
                'expected_brand': 'Samsung',
                'expected_condition': 'new',
                'description': 'Arabic query for new Samsung Note'
            },
            {
                'query': 'iPhone with good battery above 80%',
                'expected_brand': 'Apple - iPhone',
                'min_battery': 80,
                'description': 'Battery-specific requirement'
            },
            {
                'query': 'Xiaomi Redmi 8GB RAM',
                'expected_brand': 'Xiaomi',
                'expected_ram': 8,
                'description': 'RAM-specific query'
            },
            {
                'query': 'OPPO phone under 10000 EGP',
                'expected_brand': 'OPPO',
                'max_price': 10000,
                'description': 'Budget constraint query'
            },
            {
                'query': 'هاتف 256 جيجا بطارية جيدة',
                'expected_storage': 256,
                'description': 'Arabic storage and battery query'
            }
        ]

    def evaluate_query(self, test_case: Dict, top_k: int = 10) -> Dict:
        """Evaluate a single query"""
        query = test_case['query']
        
        # Perform search
        start_time = time.time()
        result = self.search.smart_search(query, top_k=top_k)
        search_time = time.time() - start_time
        
        results_df = result['results']
        filters = result['filters']
        
        evaluation = {
            'query': query,
            'description': test_case['description'],
            'search_time': search_time,
            'total_results': len(results_df),
            'filters_detected': filters,
            'filter_accuracy': 0,
            'brand_accuracy': 0,
            'constraint_satisfaction': 0,
            'relevance_scores': []
        }
        
        if len(results_df) == 0:
            return evaluation
        
        # Check filter detection accuracy
        filter_matches = 0
        total_expected_filters = 0
        
        for key in ['expected_brand', 'expected_storage', 'expected_condition', 'expected_ram', 'max_price', 'min_battery']:
            if key in test_case:
                total_expected_filters += 1
                filter_key = key.replace('expected_', '').replace('max_', 'max_').replace('min_', 'min_')
                if filter_key in filters:
                    if key.startswith('expected_'):
                        # For brand/condition, check if detected correctly
                        if key == 'expected_brand' and test_case[key].lower() in str(filters.get('brand', '')).lower():
                            filter_matches += 1
                        elif key == 'expected_condition' and test_case[key] == filters.get('condition'):
                            filter_matches += 1
                        elif key == 'expected_storage' and abs(test_case[key] - filters.get('storage_gb', 0)) <= 64:
                            filter_matches += 1
                        elif key == 'expected_ram' and test_case[key] == filters.get('ram_gb'):
                            filter_matches += 1
                    else:
                        # For price/battery constraints
                        filter_matches += 1
        
        if total_expected_filters > 0:
            evaluation['filter_accuracy'] = filter_matches / total_expected_filters
        
        # Check brand accuracy in results
        if 'expected_brand' in test_case:
            brand_matches = results_df['brand'].str.contains(test_case['expected_brand'], case=False, na=False).sum()
            evaluation['brand_accuracy'] = brand_matches / len(results_df)
        
        # Check constraint satisfaction
        constraint_score = 0
        constraint_count = 0
        
        # Storage constraint
        if 'expected_storage' in test_case:
            constraint_count += 1
            storage_matches = results_df['storage_gb'].notna() & (results_df['storage_gb'] >= test_case['expected_storage'] * 0.8)
            constraint_score += storage_matches.sum() / len(results_df)
        
        # RAM constraint
        if 'expected_ram' in test_case:
            constraint_count += 1
            ram_matches = results_df['ram_gb'].notna() & (results_df['ram_gb'] >= test_case['expected_ram'])
            constraint_score += ram_matches.sum() / len(results_df)
        
        # Price constraint
        if 'max_price' in test_case:
            constraint_count += 1
            price_matches = results_df['price_clean'].notna() & (results_df['price_clean'] <= test_case['max_price'])
            constraint_score += price_matches.sum() / len(results_df)
        
        # Battery constraint
        if 'min_battery' in test_case:
            constraint_count += 1
            battery_matches = results_df['battery_percent'].notna() & (results_df['battery_percent'] >= test_case['min_battery'])
            constraint_score += battery_matches.sum() / len(results_df)
        
        if constraint_count > 0:
            evaluation['constraint_satisfaction'] = constraint_score / constraint_count
        
        # Relevance scores
        evaluation['relevance_scores'] = results_df['similarity_score'].tolist()
        evaluation['avg_relevance'] = results_df['similarity_score'].mean()
        
        return evaluation

    def run_full_evaluation(self) -> Dict:
        """Run evaluation on all test queries"""
        print("Running comprehensive search evaluation...")
        print("=" * 60)
        
        all_evaluations = []
        
        for i, test_case in enumerate(self.test_queries):
            print(f"\nEvaluating Query {i+1}/{len(self.test_queries)}")
            print(f"Query: {test_case['query']}")
            print(f"Description: {test_case['description']}")
            
            evaluation = self.evaluate_query(test_case)
            all_evaluations.append(evaluation)
            
            # Print results
            print(f"Results: {evaluation['total_results']}")
            print(f"Search Time: {evaluation['search_time']:.3f}s")
            print(f"Filter Accuracy: {evaluation['filter_accuracy']:.2f}")
            print(f"Brand Accuracy: {evaluation['brand_accuracy']:.2f}")
            print(f"Constraint Satisfaction: {evaluation['constraint_satisfaction']:.2f}")
            print(f"Avg Relevance: {evaluation['avg_relevance']:.3f}")
            print(f"Filters Detected: {evaluation['filters_detected']}")
            print("-" * 40)
        
        # Overall statistics
        print("\n" + "=" * 60)
        print("OVERALL EVALUATION RESULTS")
        print("=" * 60)
        
        overall_stats = {
            'total_queries': len(all_evaluations),
            'avg_search_time': np.mean([e['search_time'] for e in all_evaluations]),
            'avg_filter_accuracy': np.mean([e['filter_accuracy'] for e in all_evaluations]),
            'avg_brand_accuracy': np.mean([e['brand_accuracy'] for e in all_evaluations]),
            'avg_constraint_satisfaction': np.mean([e['constraint_satisfaction'] for e in all_evaluations]),
            'avg_relevance': np.mean([e['avg_relevance'] for e in all_evaluations]),
            'successful_queries': len([e for e in all_evaluations if e['total_results'] > 0])
        }
        
        print(f"Queries Evaluated: {overall_stats['total_queries']}")
        print(f"Successful Queries: {overall_stats['successful_queries']} ({overall_stats['successful_queries']/overall_stats['total_queries']*100:.1f}%)")
        print(f"Average Search Time: {overall_stats['avg_search_time']:.3f}s")
        print(f"Average Filter Accuracy: {overall_stats['avg_filter_accuracy']:.3f}")
        print(f"Average Brand Accuracy: {overall_stats['avg_brand_accuracy']:.3f}")
        print(f"Average Constraint Satisfaction: {overall_stats['avg_constraint_satisfaction']:.3f}")
        print(f"Average Relevance Score: {overall_stats['avg_relevance']:.3f}")
        
        # Language performance
        arabic_queries = [e for e in all_evaluations if any(ord(c) > 127 for c in e['query'])]
        english_queries = [e for e in all_evaluations if not any(ord(c) > 127 for c in e['query'])]
        
        if arabic_queries:
            arabic_relevance = np.mean([e['avg_relevance'] for e in arabic_queries])
            print(f"Arabic Query Performance: {arabic_relevance:.3f}")
        
        if english_queries:
            english_relevance = np.mean([e['avg_relevance'] for e in english_queries])
            print(f"English Query Performance: {english_relevance:.3f}")
        
        return {
            'individual_evaluations': all_evaluations,
            'overall_stats': overall_stats
        }

    def demonstrate_search_comparison(self):
        """Demonstrate search quality with examples"""
        print("\n" + "=" * 60)
        print("SEARCH QUALITY DEMONSTRATION")
        print("=" * 60)
        
        demo_queries = [
            "iPhone 13 under 15k with 128GB storage",
            "أيفون استعمال خفيف 256 جيجا",
            "Samsung A52 good battery life"
        ]
        
        for query in demo_queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 40)
            
            result = self.search.smart_search(query, top_k=3)
            
            print(f"Filters detected: {result['filters']}")
            print(f"Results found: {result['total_found']}")
            
            print("\nTop 3 results:")
            for i, (_, row) in enumerate(result['results'].iterrows()):
                print(f"{i+1}. {row['title'][:60]}...")
                print(f"   Price: {row['price']} | Storage: {row['storage_gb']}GB | Battery: {row['battery_percent']}%")
                print(f"   Relevance: {row['similarity_score']:.3f}")

if __name__ == "__main__":
    evaluator = SearchEvaluator()
    
    # Run full evaluation
    results = evaluator.run_full_evaluation()
    
    # Demonstrate search quality
    evaluator.demonstrate_search_comparison()
