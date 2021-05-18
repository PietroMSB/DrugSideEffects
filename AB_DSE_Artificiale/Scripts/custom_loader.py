from spektral.data.loaders import BatchLoader

class CustomLoader(BatchLoader):
    def collate(self, batch):
        import numpy as np
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        X = pad_sequences([g.x for g in batch], padding='post', dtype=np.float32)
        Y = pad_sequences([g.y for g in batch], padding='post', dtype=np.float32)
        #E = pad_sequences([g.e for g in batch], padding='post')
        n_max = max(g.n_nodes for g in batch)
        A = np.zeros((len(batch), n_max, n_max), dtype=np.float32)
        for a,g in zip(A, batch): a[:g.n_nodes, :g.n_nodes] = g.a.toarray()
        return (X, A), Y