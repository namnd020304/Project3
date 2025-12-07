import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import zipfile
import os


class MovieLensSVD:
    """
    SVD Collaborative Filtering cho MovieLens 32M Dataset
    Tối ưu hóa cho dữ liệu lớn với sparse matrix
    """

    def __init__(self, n_factors=50, random_state=42):
        """
        Parameters:
        -----------
        n_factors : int
            Số lượng latent factors
        random_state : int
            Seed cho reproducibility
        """
        self.n_factors = n_factors
        self.random_state = random_state
        self.user_mapper = None
        self.movie_mapper = None
        self.user_inv_mapper = None
        self.movie_inv_mapper = None
        self.user_bias = None
        self.movie_bias = None
        self.global_mean = None
        self.U = None
        self.sigma = None
        self.Vt = None

    def load_movielens_data(self, data_path='ml-32m', sample_frac=None):
        """
        Load MovieLens dataset

        Parameters:
        -----------
        data_path : str
            Đường dẫn đến folder chứa dữ liệu MovieLens
        sample_frac : float (0-1)
            Sample một phần dữ liệu để test nhanh (None = dùng toàn bộ)

        Returns:
        --------
        DataFrame với columns: userId, movieId, rating, timestamp
        """
        ratings_file = os.path.join(data_path, 'ratings.csv')

        print(f"Đang load dữ liệu từ {ratings_file}...")
        start_time = time.time()

        # Load ratings
        df = pd.read_csv(ratings_file)

        if sample_frac and sample_frac < 1.0:
            print(f"Sampling {sample_frac * 100}% dữ liệu...")
            df = df.sample(frac=sample_frac, random_state=self.random_state)

        print(f"Load xong trong {time.time() - start_time:.2f}s")
        print(f"Tổng số ratings: {len(df):,}")
        print(f"Số users: {df['userId'].nunique():,}")
        print(f"Số movies: {df['movieId'].nunique():,}")
        print(f"Sparsity: {100 * (1 - len(df) / (df['userId'].nunique() * df['movieId'].nunique())):.4f}%")

        return df

    def create_user_movie_matrix(self, df):
        """
        Tạo user-movie sparse matrix từ DataFrame

        Returns:
        --------
        sparse_matrix : scipy.sparse.csr_matrix
            Ma trận ratings dạng sparse
        """
        print("\nTạo user-movie matrix...")

        # Tạo mapping từ original IDs sang continuous indices
        unique_users = df['userId'].unique()
        unique_movies = df['movieId'].unique()

        self.user_mapper = {user: idx for idx, user in enumerate(unique_users)}
        self.movie_mapper = {movie: idx for idx, movie in enumerate(unique_movies)}
        self.user_inv_mapper = {idx: user for user, idx in self.user_mapper.items()}
        self.movie_inv_mapper = {idx: movie for movie, idx in self.movie_mapper.items()}

        # Map IDs
        user_indices = df['userId'].map(self.user_mapper)
        movie_indices = df['movieId'].map(self.movie_mapper)
        ratings = df['rating'].values

        # Tạo sparse matrix
        n_users = len(unique_users)
        n_movies = len(unique_movies)

        sparse_matrix = csr_matrix(
            (ratings, (user_indices, movie_indices)),
            shape=(n_users, n_movies)
        )

        print(f"Ma trận shape: {sparse_matrix.shape}")
        print(f"Non-zero elements: {sparse_matrix.nnz:,}")

        return sparse_matrix

    def fit(self, df):
        """
        Train SVD model

        Parameters:
        -----------
        df : DataFrame
            DataFrame với columns: userId, movieId, rating
        """
        print("\n" + "=" * 60)
        print("BẮT ĐẦU TRAIN MODEL SVD")
        print("=" * 60)

        start_time = time.time()

        # Tạo sparse matrix
        R = self.create_user_movie_matrix(df)

        # Tính global mean (chỉ tính từ non-zero ratings)
        self.global_mean = df['rating'].mean()
        print(f"\nGlobal mean rating: {self.global_mean:.3f}")

        # Tính user bias và movie bias
        print("\nTính user & movie bias...")
        user_ratings = df.groupby('userId')['rating'].agg(['mean', 'count'])
        movie_ratings = df.groupby('movieId')['rating'].agg(['mean', 'count'])

        self.user_bias = {}
        for user_id, row in user_ratings.iterrows():
            self.user_bias[self.user_mapper[user_id]] = row['mean'] - self.global_mean

        self.movie_bias = {}
        for movie_id, row in movie_ratings.iterrows():
            self.movie_bias[self.movie_mapper[movie_id]] = row['mean'] - self.global_mean

        # Normalize matrix bằng cách trừ user bias
        print("Normalize matrix...")
        R_normalized = R.copy().astype(float)
        for i in range(R.shape[0]):
            if i in self.user_bias:
                # Chỉ trừ bias cho các non-zero entries
                nonzero_cols = R[i].nonzero()[1]
                for j in nonzero_cols:
                    R_normalized[i, j] -= (self.global_mean + self.user_bias.get(i, 0) +
                                           self.movie_bias.get(j, 0))

        # Thực hiện SVD
        print(f"\nThực hiện SVD với {self.n_factors} factors...")
        print("(Quá trình này có thể mất vài phút với dữ liệu lớn...)")

        svd_start = time.time()
        self.U, sigma, self.Vt = svds(R_normalized, k=self.n_factors)
        self.sigma = np.diag(sigma)

        print(f"SVD hoàn thành trong {time.time() - svd_start:.2f}s")

        print(f"\nTổng thời gian train: {time.time() - start_time:.2f}s")
        print("=" * 60)

        return self

    def predict_rating(self, user_id, movie_id):
        """
        Dự đoán rating cho một user-movie cụ thể

        Parameters:
        -----------
        user_id : int
            Original user ID
        movie_id : int
            Original movie ID

        Returns:
        --------
        float : predicted rating (clamped 0.5-5.0)
        """
        # Map sang internal indices
        if user_id not in self.user_mapper or movie_id not in self.movie_mapper:
            return self.global_mean

        user_idx = self.user_mapper[user_id]
        movie_idx = self.movie_mapper[movie_id]

        # Prediction = global_mean + user_bias + movie_bias + U * Σ * V^T
        pred = self.global_mean
        pred += self.user_bias.get(user_idx, 0)
        pred += self.movie_bias.get(movie_idx, 0)
        pred += np.dot(np.dot(self.U[user_idx, :], self.sigma), self.Vt[:, movie_idx])

        # Clamp giữa 0.5 và 5.0
        return np.clip(pred, 0.5, 5.0)

    def recommend_movies(self, user_id, n=10, movies_df=None):
        """
        Đề xuất top N movies cho user

        Parameters:
        -----------
        user_id : int
            Original user ID
        n : int
            Số lượng movies muốn đề xuất
        movies_df : DataFrame
            DataFrame chứa thông tin movies (movieId, title, genres)

        Returns:
        --------
        list : [(movie_id, predicted_rating, title, genres), ...]
        """
        if user_id not in self.user_mapper:
            print(f"User {user_id} không có trong training data")
            return []

        user_idx = self.user_mapper[user_id]

        # Tính predictions cho tất cả movies
        predictions = []
        for movie_idx in range(len(self.movie_inv_mapper)):
            movie_id = self.movie_inv_mapper[movie_idx]
            pred_rating = self.predict_rating(user_id, movie_id)
            predictions.append((movie_id, pred_rating))

        # Sort theo rating giảm dần
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Thêm thông tin movie nếu có
        results = []
        if movies_df is not None:
            movies_info = movies_df.set_index('movieId')
            for movie_id, rating in predictions[:n]:
                if movie_id in movies_info.index:
                    movie = movies_info.loc[movie_id]
                    results.append((movie_id, rating, movie['title'], movie.get('genres', 'N/A')))
                else:
                    results.append((movie_id, rating, 'Unknown', 'N/A'))
        else:
            results = predictions[:n]

        return results

    def evaluate(self, test_df):
        """
        Đánh giá model trên test set

        Parameters:
        -----------
        test_df : DataFrame
            Test data với columns: userId, movieId, rating

        Returns:
        --------
        dict : {'rmse': float, 'mae': float}
        """
        print("\nĐánh giá model trên test set...")

        predictions = []
        actuals = []

        for _, row in test_df.iterrows():
            pred = self.predict_rating(row['userId'], row['movieId'])
            predictions.append(pred)
            actuals.append(row['rating'])

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        return {'rmse': rmse, 'mae': mae}


# ============================================================================
# DEMO VÀ HƯỚNG DẪN SỬ DỤNG
# ============================================================================

def demo_with_sample_data():
    """
    Demo với sample nhỏ để test nhanh
    """
    print("\n" + "=" * 70)
    print("DEMO VỚI SAMPLE DATA (10% của dataset)")
    print("=" * 70)

    # Khởi tạo model
    model = MovieLensSVD(n_factors=50, random_state=42)

    # Load data (sample 10% để demo nhanh)
    # Thay 'ml-32m' bằng đường dẫn thực tế của bạn
    df = model.load_movielens_data(data_path='ml-latest', sample_frac=0.1)

    # Split train/test (80/20)
    print(r"\nChia train/test set...")
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    print(f"Train set: {len(train_df):,} ratings")
    print(f"Test set: {len(test_df):,} ratings")

    # Train model
    model.fit(train_df)

    # Evaluate
    metrics = model.evaluate(test_df)

    # Demo recommendations
    print("\n" + "=" * 70)
    print("DEMO ĐỀ XUẤT PHIM")
    print("=" * 70)

    # Lấy một user ngẫu nhiên
    sample_user = train_df['userId'].iloc[0]
    print(f"\nĐề xuất top 10 phim cho User {sample_user}:")

    recommendations = model.recommend_movies(sample_user, n=10)
    for i, (movie_id, rating) in enumerate(recommendations, 1):
        print(f"{i}. Movie ID {movie_id}: Predicted rating = {rating:.2f}")

    # Test prediction cho một user-movie cụ thể
    sample_movie = train_df['movieId'].iloc[0]
    pred = model.predict_rating(sample_user, sample_movie)
    print(f"\nDự đoán rating của User {sample_user} cho Movie {sample_movie}: {pred:.2f}")

    return model


if __name__ == "__main__":
   # Load toàn bộ dataset (32M ratings)
   model = MovieLensSVD(n_factors=100)
   df = model.load_movielens_data(data_path='ml-latest')

   # Hoặc sample để test nhanh
   df = model.load_movielens_data(data_path='ml-latest', sample_frac=0.1)

   # Train model
   model.fit(df)

   # Dự đoán rating
   rating = model.predict_rating(user_id=1, movie_id=1234)

   # Đề xuất phim
   recommendations = model.recommend_movies(user_id=1, n=10)

   # Sau khi train xong, thêm:

   # 1. Đánh giá model trên test set
   print("\n" + "=" * 60)
   print("ĐÁNH GIÁ MODEL")
   print("=" * 60)

   # Split một phần data để test
   test_sample = df.sample(n=10000, random_state=42)  # Lấy 10k ratings để test
   metrics = model.evaluate(test_sample)

   # 2. Thử dự đoán một số ratings cụ thể
   print("\n" + "=" * 60)
   print("THỬ DỰ ĐOÁN VÀI RATINGS")
   print("=" * 60)

   for i in range(5):
       row = df.iloc[i]
       user_id = row['userId']
       movie_id = row['movieId']
       actual = row['rating']
       predicted = model.predict_rating(user_id, movie_id)

       print(f"User {user_id} - Movie {movie_id}:")
       print(f"  Actual: {actual:.1f} | Predicted: {predicted:.2f} | Error: {abs(actual - predicted):.2f}")

   # 3. Đề xuất phim cho một user
   print("\n" + "=" * 60)
   print("ĐỀ XUẤT PHIM CHO USER")
   print("=" * 60)

   # Lấy một user có nhiều ratings
   user_counts = df['userId'].value_counts()
   active_user = user_counts.index[0]  # User active nhất

   print(f"\nTop 10 phim đề xuất cho User {active_user}:")
   recommendations = model.recommend_movies(active_user, n=10)

   for i, (movie_id, rating) in enumerate(recommendations, 1):
       print(f"{i:2d}. Movie ID {movie_id:6d} - Predicted rating: {rating:.3f}")

   # 4. Xem user đã rating những phim nào
   print(f"\nPhim mà User {active_user} đã rating:")
   user_ratings = df[df['userId'] == active_user].sort_values('rating', ascending=False).head(10)
   for _, row in user_ratings.iterrows():
       print(f"  Movie {int(row['movieId']):6d}: {row['rating']:.1f} ⭐")