from multiprocessing import Pool, cpu_count
from functools import reduce


def crawl(doc, prefix=''):
    columns = set()
    for field in doc.keys():
        if not isinstance(doc[field], list):
            columns.add(prefix + field)
        else:
            if len(doc[field]) > 0:
                for item in doc[field]:
                    if isinstance(item, dict):
                        nested_columns = crawl(item, prefix=f"{prefix}{field}_")
                        columns.update(nested_columns)
    return columns


def process_batch(docs):
    """Xử lý một batch documents"""
    all_columns = set()
    for doc in docs:
        all_columns.update(crawl(doc))
    return all_columns


def count_all_fields_parallel(collection, batch_size=10000):
    # Đếm tổng số documents
    total_docs = collection.count_documents({})
    print(f"Tổng số documents: {total_docs:,}")

    all_columns = set()
    processed = 0

    # Số lượng CPU cores
    num_processes = cpu_count()

    with Pool(processes=num_processes) as pool:
        # Lấy documents theo batch
        cursor = collection.find({}, batch_size=batch_size)

        batch = []
        batches_to_process = []

        for doc in cursor:
            batch.append(doc)

            if len(batch) >= batch_size:
                batches_to_process.append(batch)
                batch = []

                # Xử lý song song mỗi khi có đủ batches
                if len(batches_to_process) >= num_processes:
                    results = pool.map(process_batch, batches_to_process)
                    for result in results:
                        all_columns.update(result)

                    processed += sum(len(b) for b in batches_to_process)
                    print(f"Đã xử lý: {processed:,}/{total_docs:,} ({processed / total_docs * 100:.1f}%)")
                    batches_to_process = []

        # Xử lý batch cuối
        if batch:
            batches_to_process.append(batch)

        if batches_to_process:
            results = pool.map(process_batch, batches_to_process)
            for result in results:
                all_columns.update(result)

    return all_columns

import pymongo
db = "glamira"
table = "summary"
uri ="mongodb://34.57.213.13/"
client = pymongo.MongoClient(uri)
glamira = client[db]
summary = glamira[table]
# Sử dụng
all_fields = count_all_fields_parallel(summary)
print(f"\nTổng số trường: {len(all_fields)}")