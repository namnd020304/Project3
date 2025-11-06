import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gc
import os
import pickle
from functools import lru_cache
from typing import Optional, List, Union
import warnings

class HybridContentRecommender:
    """
    Kết hợp 2 phương pháp:
    1. Genre-based (như code cũ của bạn)
    2. Tag-based TF-IDF
    
    OPTIMIZED: Không tính toàn bộ similarity matrix trước
    IMPROVED: Vectorized operations, caching, persistence, batch processing
    """
    
    def __init__(self, movies_path: str, ratings_path: str, tags_path: str):
        """
        Initialize the recommender with data validation
        """
        # Validate file paths
        for path, name in [(movies_path, "movies"), (ratings_path, "ratings"), (tags_path, "tags")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} file not found: {path}")
        
        # Load data with error handling
        try:
            self.movies = pd.read_csv(movies_path)
            self.ratings = pd.read_csv(ratings_path)
            self.tags = pd.read_csv(tags_path)
        except Exception as e:
            raise ValueError(f"Error loading data files: {str(e)}")
        
        # Validate required columns
        required_movie_cols = ['movieId', 'title', 'genres']
        required_rating_cols = ['userId', 'movieId', 'rating']
        required_tag_cols = ['movieId', 'tag']
        
        missing_movie = set(required_movie_cols) - set(self.movies.columns)
        missing_rating = set(required_rating_cols) - set(self.ratings.columns)
        missing_tag = set(required_tag_cols) - set(self.tags.columns)
        
        if missing_movie or missing_rating or missing_tag:
            raise ValueError(
                f"Missing required columns - "
                f"Movies: {missing_movie}, "
                f"Ratings: {missing_rating}, "
                f"Tags: {missing_tag}"
            )
        
        # Clean data
        self.ratings = self.ratings.dropna(subset=['userId', 'movieId', 'rating'])
        self.movies = self.movies.dropna(subset=['movieId', 'title'])
        
        # Create movie_id to position mapping for consistent indexing
        self.movie_id_to_pos = pd.Series(
            index=self.movies['movieId'].values,
            data=range(len(self.movies))
        ).to_dict()
        
        # Matrices - CHỈ LƯU features, KHÔNG lưu similarity matrix
        self.genre_matrix = None
        self.tfidf_matrix = None
        self.genre_cols = None
        self.tfidf_vectorizer = None
        
        # Caching
        self._similarity_cache = {}
        self._cache_max_size = 1000  # Cache up to 1000 movie similarities
        
    def prepare_genre_features(self):
        """
        PHẦN 1: Genre-based (OPTIMIZED with vectorized operations)
        """
        print("Building Genre-based features...")
        
        # Validate genres column
        if 'genres' not in self.movies.columns:
            raise ValueError("Movies dataframe missing 'genres' column")
        
        # Fill NaN genres
        self.movies['genres'] = self.movies['genres'].fillna('(no genres listed)')
        
        # Define genre columns
        self.genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                           'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance',
                           'Sci-Fi', 'Thriller', 'War', 'Western']
        
        # VECTORIZED: Create genre columns efficiently
        # Initialize all genre columns to 0
        for col in self.genre_cols:
            self.movies[col] = 0.0
        
        # Vectorized genre encoding using str.get_dummies (much faster than iterrows)
        # Split genres and create boolean matrix
        genre_split = self.movies['genres'].str.split('|', expand=False)
        
        # Process each genre efficiently
        for genre in self.genre_cols:
            if genre in self.genre_cols:
                # Vectorized check: True if genre appears in genres_list
                mask = genre_split.apply(
                    lambda x: genre in x if isinstance(x, list) else False
                )
                # Exclude "(no genres listed)"
                no_genre_mask = self.movies['genres'] == "(no genres listed)"
                self.movies.loc[mask & ~no_genre_mask, genre] = 1.0
        
        # Ensure all genre columns exist
        for col in self.genre_cols:
            if col not in self.movies.columns:
                self.movies[col] = 0.0
            else:
                self.movies[col] = self.movies[col].fillna(0.0).astype(float)
        
        # Genre matrix
        self.genre_matrix = self.movies[self.genre_cols].values.astype(np.float32)
        
        print(f"Genre matrix shape: {self.genre_matrix.shape}")
        print(f"Memory: {self.genre_matrix.nbytes / 1024 / 1024:.2f} MB")
        print("Note: Similarity will be computed on-demand to save memory")
        
        return self.genre_cols
    
    def prepare_tag_features(self):
        """
        PHẦN 2: Tag-based TF-IDF (OPTIMIZED)
        """
        print("\nBuilding Tag-based features...")
        
        # Validate tags dataframe
        if len(self.tags) == 0:
            warnings.warn("Tags dataframe is empty. Tag-based features will be minimal.")
            self.movies['tags'] = ''
            self.movies['combined_features'] = ''
        else:
            # Clean tags - vectorized
            print("Cleaning tags...")
            self.tags['tag_clean'] = (
                self.tags['tag']
                .astype(str)
                .str.lower()
                .str.replace(r'[^a-z0-9\s]', '', regex=True)
                .str.strip()
            )
            
            # Filter tags
            print("Filtering noisy tags...")
            tag_counts = self.tags['tag_clean'].value_counts()
            valid_tags = tag_counts[tag_counts >= 3].index
            self.tags = self.tags[self.tags['tag_clean'].isin(valid_tags)]
            
            # Aggregate tags by movie - vectorized groupby
            print("Aggregating tags by movie...")
            movie_tags = self.tags.groupby('movieId')['tag_clean'].apply(
                lambda x: ' '.join(x)
            ).reset_index()
            movie_tags.columns = ['movieId', 'tags']
            
            # Merge
            self.movies = self.movies.merge(movie_tags, on='movieId', how='left')
            self.movies['tags'] = self.movies['tags'].fillna('')
        
        # Prepare genres for TF-IDF - vectorized
        self.movies['genres_clean'] = (
            self.movies['genres']
            .astype(str)
            .str.replace('|', ' ')
            .str.lower()
            .str.replace('-', '')
        )
        
        # Combine: tags (weight=3) + genres (weight=2) - vectorized
        print("Creating combined features...")
        def combine_features_vectorized(row):
            features = []
            if pd.notna(row['tags']) and row['tags']:
                features.extend([str(row['tags'])] * 3)
            if pd.notna(row['genres_clean']) and row['genres_clean']:
                features.extend([str(row['genres_clean'])] * 2)
            return ' '.join(features) if features else ''
        
        self.movies['combined_features'] = self.movies.apply(combine_features_vectorized, axis=1)
        
        # TF-IDF
        print("Computing TF-IDF...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.7,
            stop_words='english'
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movies['combined_features'].fillna('')
        )
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Memory: {self.tfidf_matrix.data.nbytes / 1024 / 1024:.2f} MB (sparse)")
        
        # Tag coverage
        has_tags = (self.movies['tags'] != '').sum()
        coverage = has_tags / len(self.movies) * 100
        print(f"Tag coverage: {coverage:.1f}% ({has_tags}/{len(self.movies)})")
        
        # Clean up
        gc.collect()
        
        return coverage
    
    def get_adaptive_weights(self, movie_idx: int) -> tuple:
        """
        Tính trọng số động dựa trên tag availability
        """
        movie = self.movies.iloc[movie_idx]
        tag_count = len(str(movie['tags']).split()) if pd.notna(movie['tags']) and movie['tags'] else 0
        
        if tag_count >= 10:
            tag_weight = 0.7
            genre_weight = 0.3
        elif tag_count >= 3:
            tag_weight = 0.3 + (tag_count - 3) * (0.4 / 7)
            genre_weight = 1 - tag_weight
        else:
            tag_weight = 0.2
            genre_weight = 0.8
        
        return genre_weight, tag_weight
    
    def compute_similarity_for_movie(self, movie_idx: int, use_cache: bool = True):
        """
        TÍNH SIMILARITY CHO 1 PHIM (on-demand)
        Không tính toàn bộ matrix
        OPTIMIZED: Added caching support
        """
        # Check cache
        if use_cache and movie_idx in self._similarity_cache:
            return self._similarity_cache[movie_idx]
        
        if self.genre_matrix is None or self.tfidf_matrix is None:
            raise ValueError("Features not prepared. Call prepare_genre_features() and prepare_tag_features() first.")
        
        # Validate index
        if movie_idx < 0 or movie_idx >= len(self.movies):
            raise IndexError(f"Movie index {movie_idx} out of range [0, {len(self.movies)})")
        
        # Genre similarity - optimized (no need for slicing)
        genre_vec = self.genre_matrix[movie_idx:movie_idx+1]
        genre_scores = cosine_similarity(genre_vec, self.genre_matrix)[0]
        
        # Tag similarity
        tag_vec = self.tfidf_matrix[movie_idx:movie_idx+1]
        tag_scores = cosine_similarity(tag_vec, self.tfidf_matrix)[0]
        
        # Adaptive weights
        genre_weight, tag_weight = self.get_adaptive_weights(movie_idx)
        
        # Hybrid
        hybrid_scores = genre_weight * genre_scores + tag_weight * tag_scores
        
        result = (hybrid_scores, genre_weight, tag_weight)
        
        # Cache result (with size limit)
        if use_cache:
            if len(self._similarity_cache) >= self._cache_max_size:
                # Remove oldest entry (simple FIFO)
                self._similarity_cache.pop(next(iter(self._similarity_cache)))
            self._similarity_cache[movie_idx] = result
        
        return result
    
    def recommend_similar_movies(self, title: str, top_n: int = 10, use_cache: bool = True):
        """
        Gợi ý phim tương tự (content-based thuần)
        IMPROVED: Better error handling and validation
        """
        if not isinstance(title, str) or not title.strip():
            return f"Invalid title: '{title}'"
        
        if self.genre_matrix is None or self.tfidf_matrix is None:
            return "Error: Features not prepared. Call prepare_genre_features() and prepare_tag_features() first."
        
        try:
            # Tìm movie (xử lý cả trường hợp có/không có năm)
            title_clean = title.strip()
            matches = self.movies[
                self.movies['title'].str.contains(title_clean, case=False, na=False, regex=False)
            ]
            
            if len(matches) == 0:
                return f"Movie '{title}' not found."
            
            idx = matches.index[0]
            movie_title = self.movies.iloc[idx]['title']
            
            # Tính similarity (chỉ cho phim này)
            print(f"Computing similarities for '{movie_title}'...")
            hybrid_scores, g_weight, t_weight = self.compute_similarity_for_movie(idx, use_cache=use_cache)
            
            # Sort
            sim_indices = hybrid_scores.argsort()[::-1][1:top_n+1]
            
            # Validate we have enough results
            if len(sim_indices) == 0:
                return f"No similar movies found for '{movie_title}'"
            
            # Results
            results = self.movies.iloc[sim_indices][
                ['movieId', 'title', 'genres']
            ].copy()
            results['similarity'] = hybrid_scores[sim_indices]
            results['rank'] = range(1, len(results) + 1)
            
            print(f"\n{'='*70}")
            print(f"RECOMMENDATIONS FOR: {movie_title}")
            print(f"{'='*70}")
            print(f"Weights: Genre={g_weight:.2f}, Tag={t_weight:.2f}")
            print(f"(Adaptive based on tag availability)\n")
            
            return results
            
        except IndexError:
            return f"Movie '{title}' not found."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def build_user_profile_genre(self, user_id: int) -> Optional[np.ndarray]:
        """
        User profile từ genres (OPTIMIZED: vectorized operations)
        """
        if not isinstance(user_id, (int, np.integer)):
            raise ValueError(f"Invalid user_id: {user_id}")
        
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return None
        
        # Merge với movies
        user_profile = user_ratings.merge(
            self.movies[['movieId'] + self.genre_cols], 
            on='movieId', 
            how='left'
        )
        
        # VECTORIZED: Weighted genres
        rating_values = user_profile['rating'].values.reshape(-1, 1)
        genre_values = user_profile[self.genre_cols].values
        
        # Element-wise multiplication
        weighted_genres = genre_values * rating_values
        
        # Sum and normalize
        genre_sums = weighted_genres.sum(axis=0)
        total = genre_sums.sum()
        
        if total > 0:
            genre_profile = genre_sums / total
        else:
            genre_profile = np.zeros(len(self.genre_cols))
        
        return genre_profile.astype(np.float32)
    
    def build_user_profile_tag(self, user_id: int) -> Optional[np.ndarray]:
        """
        User profile từ tags (TF-IDF) (OPTIMIZED: better index handling)
        """
        if not isinstance(user_id, (int, np.integer)):
            raise ValueError(f"Invalid user_id: {user_id}")
        
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return None
        
        # Lấy phim rating cao
        high_rated = user_ratings[user_ratings['rating'] >= 4.0]
        
        if len(high_rated) == 0:
            high_rated = user_ratings.nlargest(min(10, len(user_ratings)), 'rating')
        
        # OPTIMIZED: Use movie_id_to_pos mapping instead of searching
        user_vector = np.zeros(self.tfidf_matrix.shape[1], dtype=np.float32)
        total_weight = 0.0
        
        for movie_id, rating in zip(high_rated['movieId'], high_rated['rating']):
            # Use mapping for O(1) lookup instead of O(n) search
            if movie_id in self.movie_id_to_pos:
                idx = self.movie_id_to_pos[movie_id]
                user_vector += self.tfidf_matrix[idx].toarray()[0].astype(np.float32) * float(rating)
                total_weight += float(rating)
        
        if total_weight > 0:
            user_vector = user_vector / total_weight
        
        return user_vector
    
    def recommend_for_user(self, user_id: int, top_n: int = 10, exclude_watched: bool = True, verbose: bool = True):
        """
        HYBRID RECOMMENDATION cho user (OPTIMIZED)
        """
        if verbose:
            print(f"\nGenerating recommendations for User {user_id}...")
        
        # Build profiles
        genre_profile = self.build_user_profile_genre(user_id)
        tag_profile = self.build_user_profile_tag(user_id)
        
        if genre_profile is None:
            return "User not found or has no ratings."
        
        if self.genre_matrix is None or self.tfidf_matrix is None:
            return "Error: Features not prepared. Call prepare_genre_features() and prepare_tag_features() first."
        
        # Genre-based scores
        if verbose:
            print("Computing genre-based scores...")
        genre_scores = self.genre_matrix.dot(genre_profile)
        
        # Tag-based scores
        if verbose:
            print("Computing tag-based scores...")
        if tag_profile is not None:
            tag_scores = cosine_similarity([tag_profile], self.tfidf_matrix)[0]
        else:
            tag_scores = np.zeros(len(self.movies))
        
        # Adaptive weights
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        # OPTIMIZED: Single pass for watched movies and tag counts
        watched_movie_ids = set(user_ratings['movieId'].values)
        watched_mask = self.movies['movieId'].isin(watched_movie_ids)
        
        if watched_mask.any():
            watched_tags = self.movies.loc[watched_mask, 'tags']
            avg_tag_count = watched_tags.apply(
                lambda x: len(str(x).split()) if pd.notna(x) and x else 0
            ).mean()
        else:
            avg_tag_count = 0
        
        if avg_tag_count >= 5:
            genre_weight, tag_weight = 0.3, 0.7
        elif avg_tag_count >= 2:
            genre_weight, tag_weight = 0.5, 0.5
        else:
            genre_weight, tag_weight = 0.7, 0.3
        
        # Hybrid scores
        hybrid_scores = genre_weight * genre_scores + tag_weight * tag_scores
        
        # Exclude watched
        if exclude_watched:
            hybrid_scores[watched_mask.values] = -1
        
        # Top N
        top_indices = hybrid_scores.argsort()[::-1][:top_n]
        
        # Filter out excluded (if all top N are excluded, we'll get negative scores)
        valid_indices = top_indices[hybrid_scores[top_indices] >= 0]
        
        if len(valid_indices) == 0:
            return f"No recommendations available for user {user_id} (all movies already watched)"
        
        # Results
        results = self.movies.iloc[valid_indices][
            ['movieId', 'title', 'genres']
        ].copy()
        results['score'] = hybrid_scores[valid_indices]
        results['rank'] = range(1, len(results) + 1)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"RECOMMENDATIONS FOR USER {user_id}")
            print(f"{'='*70}")
            print(f"Weights: Genre={genre_weight:.2f}, Tag={tag_weight:.2f}")
            print(f"(Based on avg {avg_tag_count:.1f} tags/movie in user history)\n")
        
        return results
    
    def recommend_for_users_batch(self, user_ids: List[int], top_n: int = 10, exclude_watched: bool = True) -> dict:
        """
        Batch processing for multiple users (NEW FEATURE)
        Returns: {user_id: recommendations_dataframe}
        """
        results = {}
        for user_id in user_ids:
            try:
                recs = self.recommend_for_user(user_id, top_n=top_n, exclude_watched=exclude_watched, verbose=False)
                if not isinstance(recs, str):  # Not an error message
                    results[user_id] = recs
            except Exception as e:
                results[user_id] = f"Error: {str(e)}"
        return results
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk (NEW FEATURE)
        """
        if self.genre_matrix is None or self.tfidf_matrix is None:
            raise ValueError("Model not trained. Prepare features first.")
        
        model_data = {
            'genre_matrix': self.genre_matrix,
            'tfidf_matrix': self.tfidf_matrix,
            'genre_cols': self.genre_cols,
            'movies': self.movies,
            'ratings': self.ratings,
            'movie_id_to_pos': self.movie_id_to_pos,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, ratings_path: Optional[str] = None):
        """
        Load a trained model from disk (NEW FEATURE)
        Note: ratings_path is optional since ratings may be updated separately
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls.__new__(cls)
        instance.genre_matrix = model_data['genre_matrix']
        instance.tfidf_matrix = model_data['tfidf_matrix']
        instance.genre_cols = model_data['genre_cols']
        instance.movies = model_data['movies']
        instance.movie_id_to_pos = model_data['movie_id_to_pos']
        instance.tfidf_vectorizer = model_data['tfidf_vectorizer']
        
        # Load ratings (from file or use saved)
        if ratings_path:
            instance.ratings = pd.read_csv(ratings_path)
        else:
            instance.ratings = model_data.get('ratings', pd.DataFrame())
        
        # Initialize cache
        instance._similarity_cache = {}
        instance._cache_max_size = 1000
        
        # Reconstruct tags from movies (if needed)
        if 'tags' in instance.movies.columns:
            instance.tags = pd.DataFrame({
                'movieId': instance.movies['movieId'],
                'tag': instance.movies['tags']
            })
        else:
            instance.tags = pd.DataFrame(columns=['movieId', 'tag'])
        
        print(f"Model loaded from {filepath}")
        return instance
    
    def clear_cache(self):
        """Clear similarity cache"""
        self._similarity_cache.clear()
        print("Cache cleared")


# ===== USAGE =====
if __name__ == "__main__":
    import time
    
    print("="*70)
    print("HYBRID CONTENT-BASED RECOMMENDATION SYSTEM")
    print("Optimized for large datasets (87K+ movies)")
    print("="*70)
    
    # Initialize
    start = time.time()
    recommender = HybridContentRecommender(
        movies_path="E:\Project2\ml-32m\ml-latest\movies.csv",
        ratings_path="E:\Project2\ml-32m\ml-latest\\ratings.csv",
        tags_path="E:\Project2\ml-32m\ml-latest\\tags.csv"
    )
    
    # Build features
    print("\nStep 1: Building Genre features...")
    recommender.prepare_genre_features()
    
    print("\nStep 2: Building Tag features...")
    recommender.prepare_tag_features()
    
    build_time = time.time() - start
    print(f"\nTotal build time: {build_time:.2f} seconds")
    
    # Save model (optional)
    # recommender.save_model("recommender_model.pkl")
    
    # Test 1: Movie similarity
    print("\n" + "="*70)
    print("TEST 1: SIMILAR MOVIES")
    print("="*70)
    
    similar = recommender.recommend_similar_movies("Toy Story", top_n=10)
    if isinstance(similar, pd.DataFrame):
        print(similar.to_string(index=False))
    else:
        print(similar)
    
    # Test 2: User recommendations
    print("\n" + "="*70)
    print("TEST 2: USER RECOMMENDATIONS")
    print("="*70)
    
    user_recs = recommender.recommend_for_user(user_id=1, top_n=10)
    if isinstance(user_recs, pd.DataFrame):
        print(user_recs.to_string(index=False))
    else:
        print(user_recs)
    
    # Test 3: Batch processing (NEW)
    print("\n" + "="*70)
    print("TEST 3: BATCH PROCESSING (NEW)")
    print("="*70)
    batch_results = recommender.recommend_for_users_batch([1, 2, 3], top_n=5)
    print(f"Processed {len(batch_results)} users")
    for user_id, recs in list(batch_results.items())[:2]:  # Show first 2
        if isinstance(recs, pd.DataFrame):
            print(f"\nUser {user_id}:")
            print(recs[['title', 'score']].to_string(index=False))
    
    print("\n" + "="*70)
    print("System ready! Memory-efficient design with improvements.")
    print("="*70)
