import heapq
import sys
import time

import degrees

explored_count = 0
original_neighbors_for_person = degrees.neighbors_for_person


def counting_neighbors_for_person(person_id):
    global explored_count
    explored_count += 1
    return original_neighbors_for_person(person_id)


degrees.neighbors_for_person = counting_neighbors_for_person


def astar(source, target):
    if source == target:
        return []

    # Everyone who starred with the target (used by the heuristic)
    costars = {person for _, person in degrees.neighbors_for_person(target)}

    def heuristic(person):
        if person == target:
            return 0
        if person in costars:
            return 1
        return 2

    # Frontier is a priority queue ordered by f = cost + heuristic
    frontier = [(heuristic(source), source)]
    cost = {source: 0}          # steps from the source
    parent = {source: None}     # person -> (previous person, movie)
    explored = set()

    while frontier:
        _, person = heapq.heappop(frontier)

        if person == target:
            path = []
            while parent[person] is not None:
                previous, movie_id = parent[person]
                path.append((movie_id, person))
                person = previous
            path.reverse()
            return path

        if person in explored:
            continue
        explored.add(person)

        for movie_id, neighbor in degrees.neighbors_for_person(person):
            new_cost = cost[person] + 1
            # Only add the neighbor if this is a new or shorter way to reach it
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = (person, movie_id)
                heapq.heappush(frontier, (new_cost + heuristic(neighbor), neighbor))

    return None


def run(name, search, source, target):
    global explored_count
    explored_count = 0
    start = time.perf_counter()
    path = search(source, target)
    elapsed = (time.perf_counter() - start) * 1000

    degrees_found = "not connected" if path is None else len(path)
    print(f"{name:<5} degrees: {degrees_found}, "
          f"people explored: {explored_count}, time: {elapsed:.2f} ms")


def main():
    directory = sys.argv[1] if len(sys.argv) == 2 else "large"
    print("Loading data...")
    degrees.load_data(directory)
    print("Data loaded.")

    source = degrees.person_id_for_name(input("Name: "))
    if source is None:
        sys.exit("Person not found.")
    target = degrees.person_id_for_name(input("Name: "))
    if target is None:
        sys.exit("Person not found.")

    run("BFS", degrees.shortest_path, source, target)
    run("A*", astar, source, target)


if __name__ == "__main__":
    main()