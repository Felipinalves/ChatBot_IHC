"""Dublês em memória do Firestore e do Firebase Auth para os testes."""
import itertools

_ids = itertools.count(1)


class FakeSnapshot:
    def __init__(self, ref):
        self.reference = ref
        self.id = ref.id

    def to_dict(self):
        return dict(self.reference._data)


class FakeDocRef:
    def __init__(self, store, path, doc_id):
        self._store = store
        self._path = path
        self.id = doc_id

    @property
    def _key(self):
        return (self._path, self.id)

    @property
    def _data(self):
        return self._store.docs[self._key]

    def set(self, data):
        self._store.docs[self._key] = dict(data)

    def update(self, data):
        if self._key not in self._store.docs:
            raise KeyError(f"Documento inexistente: {self._key}")
        self._store.docs[self._key].update(data)

    def delete(self):
        self._store.docs.pop(self._key, None)

    def collection(self, name):
        return FakeQuery(self._store, f"{self._path}/{self.id}/{name}")


class FakeQuery:
    def __init__(self, store, path, filters=(), order=None, limit_=None):
        self._store = store
        self._path = path
        self._filters = filters
        self._order = order
        self._limit = limit_

    def document(self, doc_id=None):
        return FakeDocRef(self._store, self._path, doc_id or f"doc{next(_ids)}")

    def where(self, field, op, value):
        return FakeQuery(self._store, self._path, self._filters + ((field, op, value),), self._order, self._limit)

    def order_by(self, field, direction="ASCENDING"):
        return FakeQuery(self._store, self._path, self._filters, (field, direction), self._limit)

    def limit(self, n):
        return FakeQuery(self._store, self._path, self._filters, self._order, n)

    def add(self, data):
        ref = self.document()
        ref.set(data)
        return None, ref

    def stream(self):
        ops = {"==": lambda a, b: a == b, "<": lambda a, b: a < b}
        refs = [
            FakeDocRef(self._store, path, doc_id)
            for (path, doc_id) in list(self._store.docs)
            if path == self._path
        ]
        refs = [r for r in refs if all(ops[op](r._data.get(f), v) for f, op, v in self._filters)]
        if self._order:
            field, direction = self._order
            refs.sort(key=lambda r: r._data[field], reverse=direction == "DESCENDING")
        if self._limit:
            refs = refs[: self._limit]
        return [FakeSnapshot(r) for r in refs]


class FakeBatch:
    def __init__(self):
        self._ops = []

    def set(self, ref, data):
        self._ops.append(lambda: ref.set(data))

    def update(self, ref, data):
        self._ops.append(lambda: ref.update(data))

    def delete(self, ref):
        self._ops.append(ref.delete)

    def commit(self):
        for op in self._ops:
            op()
        self._ops = []


class FakeFirestore:
    def __init__(self):
        self.docs = {}

    def collection(self, name):
        return FakeQuery(self, name)

    def batch(self):
        return FakeBatch()


class FakeAuth:
    def __init__(self):
        self.current_user = None
        self.users = {"aluno@utfpr.edu.br": "senha123"}

    def sign_in_with_email_and_password(self, email, password):
        if self.users.get(email) != password:
            raise Exception(None, '{"error": {"message": "INVALID_LOGIN_CREDENTIALS"}}')
        return {"localId": "user-1", "idToken": "token"}

    def create_user_with_email_and_password(self, email, password):
        self.users[email] = password
        return {"localId": "user-2", "idToken": "token"}

    def send_password_reset_email(self, email):
        return None


class FakeFirebaseApp:
    def __init__(self):
        self._auth = FakeAuth()

    def auth(self):
        return self._auth
