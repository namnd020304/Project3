import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import time


def get_dataframe_ratings_base(filepath, file_format='csv'):
    """
    Đọc file ratings từ MovieLens
    Args:
        filepath: đường dẫn file
        file_format: 'csv' (ml-latest) hoặc 'tsv' (ml-100k)
    Returns:
        Y_data: numpy array với 3 cột [user_id, item_id, rating]
    """
    if file_format == 'tsv':
        # MovieLens 100K format
        r_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        ratings = pd.read_csv(filepath, sep='\t', names=r_cols, encoding='latin-1')
    else:
        # MovieLens Latest/32M format (CSV)
        ratings = pd.read_csv(filepath)
        ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    # Chỉ lấy 3 cột cần thiết
    Y_data = ratings[['user_id', 'item_id', 'rating']].values

    print(f"Loaded {len(Y_data):,} ratings")
    print(f"Users: {ratings['user_id'].nunique():,}")
    print(f"Items: {ratings['item_id'].nunique():,}")

    return Y_data


class CF(object):
    """
    Collaborative Filtering - IMPROVED VERSION
    Hỗ trợ cả User-User CF và Item-Item CF
    Tối ưu cho dataset lớn với sparse matrix
    """

    def __init__(self, data_matrix, k, dist_func=cosine_similarity, uuCF=1,
                 min_support=20, verbose=True):
        """
        Args:
            data_matrix: numpy array [user_id, item_id, rating]
            k: số lượng neighbors để dự đoán
            dist_func: hàm tính similarity (mặc định cosine_similarity)
            uuCF: 1 = User-User CF, 0 = Item-Item CF
            min_support: lọc users/items có ít nhất min_support ratings
            verbose: in thông tin debug
        """
        self.verbose = verbose
        self.uuCF = uuCF
        self.k = k
        self.dist_func = dist_func

        # Filter data
        if min_support > 0:
            data_matrix = self._filter_data(data_matrix, min_support)

        # Swap columns nếu Item-Item CF
        self.Y_data = data_matrix if uuCF else data_matrix[:, [1, 0, 2]]

        # Tạo mapping từ original ID → continuous index (0, 1, 2, ...)
        self._create_id_mappings()

        # Convert IDs sang indices
        self._convert_to_indices()

        # Thông tin ma trận
        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"COLLABORATIVE FILTERING - {'USER-USER' if uuCF else 'ITEM-ITEM'}")
            print(f"{'=' * 70}")
            print(f"Ratings: {len(self.Y_data_indexed):,}")
            print(f"{'Users' if uuCF else 'Items'}: {self.n_users:,}")
            print(f"{'Items' if uuCF else 'Users'}: {self.n_items:,}")
            print(f"Sparsity: {100 * (1 - len(self.Y_data_indexed) / (self.n_users * self.n_items)):.2f}%")
            print(f"k (neighbors): {k}")

        # Matrices
        self.Ybar = None
        self.Ybar_data = None
        self.mu = None
        self.S = None

    def _filter_data(self, data_matrix, min_support):
        """Lọc users và items có ít ratings"""
        if self.verbose:
            print(f"\nFiltering data (min_support={min_support})...")

        df = pd.DataFrame(data_matrix, columns=['user_id', 'item_id', 'rating'])
        original_len = len(df)

        # Iterative filtering
        for _ in range(3):
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_support].index
            df = df[df['user_id'].isin(valid_users)]

            item_counts = df['item_id'].value_counts()
            valid_items = item_counts[item_counts >= min_support].index
            df = df[df['item_id'].isin(valid_items)]

        if self.verbose:
            print(f"Retained {len(df):,} / {original_len:,} ratings ({100 * len(df) / original_len:.1f}%)")

        return df.values

    def _create_id_mappings(self):
        """Tạo mapping từ original IDs → continuous indices"""
        unique_users = np.unique(self.Y_data[:, 0])
        unique_items = np.unique(self.Y_data[:, 1])

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}

        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_item_id = {idx: iid for iid, idx in self.item_id_to_idx.items()}

        self.n_users = len(unique_users)
        self.n_items = len(unique_items)

    def _convert_to_indices(self):
        """Convert original IDs sang continuous indices"""
        self.Y_data_indexed = self.Y_data.copy()

        for i in range(len(self.Y_data)):
            self.Y_data_indexed[i, 0] = self.user_id_to_idx[self.Y_data[i, 0]]
            self.Y_data_indexed[i, 1] = self.item_id_to_idx[self.Y_data[i, 1]]

    def normalize_matrix(self):
        """
        Chuẩn hóa ma trận:
        - Tính trung bình rating của mỗi user
        - Trừ rating cho trung bình
        - Tạo sparse matrix
        """
        if self.verbose:
            print("\nNormalizing matrix...")

        start_time = time.time()

        users = self.Y_data_indexed[:, 0].astype(np.int32)
        self.Ybar_data = self.Y_data_indexed.copy()
        self.mu = np.zeros(self.n_users)

        # Tính mean cho mỗi user
        for n in range(self.n_users):
            ids = np.where(users == n)[0]
            if len(ids) > 0:
                ratings = self.Y_data_indexed[ids, 2]
                m = np.mean(ratings)
                self.mu[n] = m if not np.isnan(m) else 0
                # Chuẩn hóa
                self.Ybar_data[ids, 2] = ratings - self.mu[n]

        # Tạo sparse matrix (items x users)
        self.Ybar = sparse.coo_matrix(
            (self.Ybar_data[:, 2],
             (self.Ybar_data[:, 1].astype(np.int32),
              self.Ybar_data[:, 0].astype(np.int32))),
            shape=(self.n_items, self.n_users)
        ).tocsr()

        elapsed = time.time() - start_time
        memory_mb = (self.Ybar.data.nbytes + self.Ybar.indices.nbytes +
                     self.Ybar.indptr.nbytes) / (1024 ** 2)

        if self.verbose:
            print(f"Normalized matrix: {self.Ybar.shape}")
            print(f"Memory: {memory_mb:.2f} MB")
            print(f"Time: {elapsed:.2f}s")

    def similarity(self):
        """
        Tính similarity matrix giữa các users (hoặc items)
        """
        if self.Ybar is None:
            raise ValueError("Must call normalize_matrix() first!")

        if self.verbose:
            print("\nComputing similarity matrix...")

        start_time = time.time()

        # S[i, j] = similarity giữa user i và user j (hoặc item i và item j)
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"Similarity matrix: {self.S.shape}")
            print(f"Time: {elapsed:.2f}s")

    def _pred(self, u, i, normalized=1):
        """
        Dự đoán rating của user u cho item i

        Args:
            u: user index (continuous 0, 1, 2, ...)
            i: item index (continuous 0, 1, 2, ...)
            normalized: 1 = trả về normalized rating, 0 = trả về actual rating

        Returns:
            predicted rating
        """
        # Tìm tất cả users đã rate item i
        ids = np.where(self.Y_data_indexed[:, 1] == i)[0].astype(np.int32)

        if len(ids) == 0:
            # Item chưa được rate bởi ai
            return self.mu[u] if not normalized else 0

        users_rated_i = self.Y_data_indexed[ids, 0].astype(np.int32)

        # Lấy similarity của user u với các users đã rate item i
        sim = self.S[u, users_rated_i]

        # Lấy k neighbors gần nhất (highest similarity)
        k_actual = min(self.k, len(sim))
        a = np.argsort(sim)[-k_actual:]  # indices of top k

        nearest_s = sim[a]

        # Lấy ratings (normalized) của k neighbors
        r = self.Ybar[i, users_rated_i[a]].toarray()[0]

        # Weighted average
        denominator = np.abs(nearest_s).sum()

        if denominator < 1e-8:
            # Không có neighbor hữu ích
            return self.mu[u] if not normalized else 0

        if normalized:
            return (r * nearest_s).sum() / denominator
        else:
            return (r * nearest_s).sum() / denominator + self.mu[u]

    def pred(self, u_original, i_original, normalized=1):
        """
        Dự đoán rating (public method - nhận original IDs)

        Args:
            u_original: original user ID
            i_original: original item ID
            normalized: 1 = normalized rating, 0 = actual rating
        """
        # Convert original IDs sang indices
        if u_original not in self.user_id_to_idx:
            return None  # User không tồn tại
        if i_original not in self.item_id_to_idx:
            return None  # Item không tồn tại

        u = self.user_id_to_idx[u_original]
        i = self.item_id_to_idx[i_original]

        if self.uuCF:
            return self._pred(u, i, normalized)
        else:
            # Item-Item CF: swap u và i
            return self._pred(i, u, normalized)

    def recommend_top(self, u_original, top_n=10, return_scores=True):
        """
        Gợi ý top N items cho user

        Args:
            u_original: original user ID
            top_n: số lượng items gợi ý
            return_scores: có trả về predicted scores không

        Returns:
            list of (item_id, predicted_rating) nếu return_scores=True
            list of item_id nếu return_scores=False
        """
        if u_original not in self.user_id_to_idx:
            return []

        u = self.user_id_to_idx[u_original]

        # Items đã được rate bởi user
        ids = np.where(self.Y_data_indexed[:, 0] == u)[0]
        items_rated_by_u = set(self.Y_data_indexed[ids, 1].astype(int))

        # Dự đoán cho tất cả items chưa rate
        predictions = []

        for i in range(self.n_items):
            if i not in items_rated_by_u:
                if self.uuCF:
                    rating = self._pred(u, i, normalized=0)
                else:
                    rating = self._pred(i, u, normalized=0)

                # Convert index → original ID
                item_id = self.idx_to_item_id[i]
                predictions.append((item_id, rating))

        # Sort theo rating giảm dần
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Lấy top N
        top_items = predictions[:top_n]

        if return_scores:
            return top_items
        else:
            return [item_id for item_id, _ in top_items]

    def fit(self):
        """Train model (normalize + compute similarity)"""
        if self.verbose:
            print(f"\n{'=' * 70}")
            print("TRAINING MODEL...")
            print(f"{'=' * 70}")

        start_time = time.time()

        self.normalize_matrix()
        self.similarity()

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"\n✅ Training completed in {elapsed:.2f}s")

        return self


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    print("=" * 70)
    print("COLLABORATIVE FILTERING - IMPROVED VERSION")
    print("=" * 70)

    # Load data
    Y_data = get_dataframe_ratings_base(
        filepath="ml-latest-small/ratings.csv",  # Đường dẫn file của bạn
        file_format='csv'
    )

    # USER-USER CF
    print("\n" + "=" * 70)
    print("TRAINING USER-USER CF")
    print("=" * 70)

    cf_user = CF(
        data_matrix=Y_data,
        k=50,  # Số neighbors
        uuCF=1,  # User-User CF
        min_support=50,  # Lọc users/items có >= 50 ratings
        verbose=True
    )

    cf_user.fit()

    # Test recommendations
    print("\n" + "=" * 70)
    print("USER-USER RECOMMENDATIONS")
    print("=" * 70)

    user_id = 1  # Original user ID
    top_items = cf_user.recommend_top(user_id, top_n=10)

    print(f"\nTop 10 recommendations for User {user_id}:")
    for rank, (item_id, score) in enumerate(top_items, 1):
        print(f"  {rank}. Item {item_id}: {score:.3f}")

    # ITEM-ITEM CF
    print("\n" + "=" * 70)
    print("TRAINING ITEM-ITEM CF")
    print("=" * 70)

    cf_item = CF(
        data_matrix=Y_data,
        k=50,
        uuCF=0,  # Item-Item CF
        min_support=50,
        verbose=True
    )

    cf_item.fit()

    # Test recommendations
    print("\n" + "=" * 70)
    print("ITEM-ITEM RECOMMENDATIONS")
    print("=" * 70)

    top_items = cf_item.recommend_top(user_id, top_n=10)

    print(f"\nTop 10 recommendations for User {user_id}:")
    for rank, (item_id, score) in enumerate(top_items, 1):
        print(f"  {rank}. Item {item_id}: {score:.3f}")

    print("\n" + "=" * 70)
    print("✅ COMPLETED!")
    print("=" * 70)