"""Module to compare the performance of sorting and searching algorithms"""

import sys
from typing import List, Callable, Any
import time
import numpy as np

sys.setrecursionlimit(100000)


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to measure the time taken by a function
    """

    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        print(f"Time taken by {func.__name__}:\n\t{round(duration_ms, 4)} ms")
        return result

    return wrapper


@timer
def linear_search(nums: List[int], target: int) -> int:
    """
    Linear search for a target in a list of numbers
    """
    for i, _ in enumerate(nums):
        if nums[i] == target:
            return i
    return -1


@timer
def binary_search(nums: List[int], target: int) -> int:
    """
    Binary search for a target in a sorted list of numbers
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            # Handle duplicates by moving to the leftmost occurance
            while mid > 0 and nums[mid - 1] == target:
                mid -= 1
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


@timer
def built_in_search(nums: List[int], target: int) -> int:
    """
    Uses the built-in index() method to find the target
    Linear search but a bit faster because of implementation in C
    """
    return nums.index(target)


@timer
def built_in_sort(nums: List[int]) -> List[int]:
    """
    Uses the built-in sort() method to sort the list
    Uses TimSort which is a hybrid sorting algorithm derived from merge sort and insertion sort
    """
    return sorted(nums)


@timer
def heap_sort_all(nums: List[int]) -> List[int]:
    """
    Heap sort using max heap
    """

    def heapify(nums: List[int], n: int, i: int) -> None:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and nums[left] > nums[largest]:
            largest = left
        if right < n and nums[right] > nums[largest]:
            largest = right
        if largest != i:
            nums[i], nums[largest] = nums[largest], nums[i]
            heapify(nums, n, largest)

    def heap_sort(nums: List[int]) -> List[int]:
        n = len(nums)
        for i in range(n // 2, -1, -1):
            heapify(nums, n, i)
        for i in range(n - 1, 0, -1):
            nums[i], nums[0] = nums[0], nums[i]
            heapify(nums, i, 0)
        return nums

    return heap_sort(nums)


@timer
def merge_sort_all(nums: List[int]) -> List[int]:
    """
    Merge sort using top-down approach
    """

    def merge_sort(nums):
        """
        Recursively divides the list into two halves and merges them back
        """
        if len(nums) == 1:
            return nums
        mid = len(nums) // 2
        left = nums[: mid + 1]
        right = nums[mid + 1 :]
        result = merge(left, right)
        return result

    def merge(left, right):
        """
        Merges two sorted lists into one sorted list
        """
        result = []
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result += left[i:]
        result += right[j:]
        return result

    return merge_sort(nums)


@timer
def quick_sort_a(nums: List[int]) -> List[int]:
    """
    Quick sort using Lomuto partition scheme
    """

    def partition(nums: List[int], low: int, high: int) -> int:
        pivot = nums[high]  # select last element
        i = low - 1
        for j in range(low, high):
            if nums[j] < pivot:
                i += 1
                nums[i], nums[j] = nums[j], nums[i]
        nums[i + 1], nums[high] = nums[high], nums[i + 1]
        return i + 1

    def quick_sort_helper(nums: List[int], low: int, high: int) -> None:
        if low < high:
            pi = partition(nums, low, high)
            quick_sort_helper(nums, low, pi - 1)
            quick_sort_helper(nums, pi + 1, high)

    quick_sort_helper(nums, 0, len(nums) - 1)
    return nums


@timer
def quick_sort_b(nums: List[int]) -> List[int]:
    """
    Quick sort using Hoare partition scheme
    """

    def quick_sort(nums, is_top_level=True):
        if len(nums) > 1:
            _ = is_top_level  # suppress unused variable warning
            pivot = nums.pop()
            greater, equal, smaller = [], [pivot], []
            for item in nums:
                if item == pivot:
                    equal.append(item)
                elif item > pivot:
                    greater.append(item)
                else:
                    smaller.append(item)
            return quick_sort(smaller, False) + equal + quick_sort(greater, False)
        return nums

    return quick_sort(nums.copy())


nums_selected = np.random.randint(0, 10000, size=10000)
nums_selected = nums_selected.tolist()

heap_sort_all(nums_selected)
built_in_sort(nums_selected)
merge_sort_all(nums_selected)
quick_sort_a(nums_selected)
quick_sort_b(nums_selected)

nums_selected = sorted(nums_selected)
selected_target = np.random.choice(nums_selected)

linear_search(nums_selected, selected_target)
built_in_search(nums_selected, selected_target)
binary_search(nums_selected, selected_target)
