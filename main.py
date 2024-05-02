import time
from typing import List, Callable, Any
import numpy as np
import sys

sys.setrecursionlimit(100000)


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        print(f"Time taken by {func.__name__}:\n\t{round(duration_ms, 4)} ms")
        return result
    return wrapper


@timer
def linearSearch(nums: List[int], target: int) -> int:
    for i in range(len(nums)):
        if nums[i] == target:
            return i
    return -1


@timer
def binarySearch(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            # Handle duplicates by moving to the leftmost occurance
            while mid > 0 and nums[mid - 1] == target:
                mid -= 1
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


@timer
def builtinSearch(nums: List[int], target: int) -> int:
    """
    Uses the built-in index() method to find the target
    Linear search but a bit faster because of implementation in C
    """
    return nums.index(target)


@timer
def builtinSort(nums: List[int]) -> List[int]:
    """
    Uses the built-in sort() method to sort the list
    Uses TimSort which is a hybrid sorting algorithm derived from merge sort and insertion sort
    """
    return sorted(nums)


@timer
def heapSortAll(nums: List[int]) -> List[int]:
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

    def heapSort(nums: List[int]) -> List[int]:
        n = len(nums)
        for i in range(n // 2, -1, -1):
            heapify(nums, n, i)
        for i in range(n - 1, 0, -1):
            nums[i], nums[0] = nums[0], nums[i]
            heapify(nums, i, 0)
        return nums

    return heapSort(nums)


@timer
def mergeSortAll(nums: List[int]) -> List[int]:
    def mergeSort(nums):
        if len(nums) == 1:
            return nums
        mid = len(nums) // 2
        left = nums[:mid+1]
        right = nums[mid+1:]
        result = merge(left, right)
        return result

    def merge(left, right):
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

    return mergeSort(nums)


@timer
def quickSortA(nums: List[int]) -> List[int]:
    def partition(nums: List[int], low: int, high: int) -> int:
        pivot = nums[high]  # select last element
        i = low - 1
        for j in range(low, high):
            if nums[j] < pivot:
                i += 1
                nums[i], nums[j] = nums[j], nums[i]
        nums[i + 1], nums[high] = nums[high], nums[i + 1]
        return i + 1

    def quickSortHelper(nums: List[int], low: int, high: int) -> None:
        if low < high:
            pi = partition(nums, low, high)
            quickSortHelper(nums, low, pi - 1)
            quickSortHelper(nums, pi + 1, high)
    quickSortHelper(nums, 0, len(nums) - 1)
    return nums


@timer
def quickSortB(nums: List[int]) -> List[int]:
    def quick_sort(nums, is_top_level=True):
        if len(nums) > 1:
            pivot = nums.pop()
            greater, equal, smaller = [], [pivot], []
            for item in nums:
                if item == pivot:
                    equal.append(item)
                elif item > pivot:
                    greater.append(item)
                else:
                    smaller.append(item)
            return (quick_sort(smaller, False) +
                    equal +
                    quick_sort(greater, False))
        else:
            return nums

    return quick_sort(nums.copy())


n = 10000
nums = np.random.randint(0, n, size=n)
nums = nums.tolist()

heapSortAll(nums)
builtinSort(nums)
mergeSortAll(nums)
quickSortA(nums)
quickSortB(nums)

nums = sorted(nums)
target = np.random.choice(nums)

linearSearch(nums, target)
builtinSearch(nums, target)
binarySearch(nums, target)
