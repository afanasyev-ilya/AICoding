MATMUL_PROMPT = """import numpy as np

def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    Compute C = A @ B using explicit Python loops.
    A: shape (n, k)
    B: shape (k, m)
    Returns:
        C: shape (n, m)
    '''
    assert A.ndim == 2 and B.ndim == 2
    n, k = A.shape
    k2, m = B.shape
    assert k == k2

    C = np.zeros((n, m), dtype=A.dtype)
"""

BINARY_SEARCH_PROMPT = """import numpy as np

def binary_search(arr, target):
    '''
    Performs a binary search on a sorted list to find a target element.

    Args:
        arr: The sorted list to search within.
        target: The element to search for.

    Returns:
        The index of the target element if found, otherwise -1.
    '''
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2  # Calculate the middle index
        mid_val = arr[mid]       # Get the value at the middle index

"""

CPP_PROMPTS = [
    """
struct Coord {
    int x;
""",

    """
std::vector<int> data(n, 0);
for (int i = 0; i
""",

    """
int binary_search(const std::vector<int>& a, int target) {
    int l = 0, r = (int)a.size() - 1;
    while (l <= r) {
        int mid = l + (r - l) / 2;
""",

    """
int sum_vector(const std::vector<int>& a) {
    int s = 0;
    for (int x : a) {
""",

    """
class Point {
public:
    Point(int x, int y) : x_(x), y_(y) {}

    int getX() const {
""",

    """
std::unordered_map<std::string, int> freq;
for (const auto& s : words) {
    auto it = freq.find(s);
    if (it == freq.end()) {
""",

    """
std::vector<std::pair<int,int>> items = {/* ... */};
std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
""",

    """
template <typename T>
T my_max(const T& a, const T& b) {
""",

    """
int n;
std::cin >> n;
std::vector<int> a(n);
for (int i = 0; i < n; ++i) {
""",

    """
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int t;
    std::cin >> t;
    while (t--) {
"""
]
