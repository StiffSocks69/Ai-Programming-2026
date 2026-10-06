import random
import unittest

import compare
import degrees


def build_random_graph(rng):
    degrees.people.clear()
    degrees.movies.clear()
    for p in range(150):
        degrees.people[f"p{p}"] = {"movies": set()}
    for m in range(60):
        degrees.movies[f"m{m}"] = {"stars": set()}
        for person in rng.sample(list(degrees.people), rng.randint(1, 5)):
            degrees.movies[f"m{m}"]["stars"].add(person)
            degrees.people[person]["movies"].add(f"m{m}")


def is_valid_path(source, target, path):
    current = source
    for movie_id, person_id in path:
        stars = degrees.movies[movie_id]["stars"]
        if current not in stars or person_id not in stars:
            return False
        current = person_id
    return current == target


class TestSearch(unittest.TestCase):

    def test_astar_matches_bfs(self):
        rng = random.Random(1)
        for _ in range(30):
            build_random_graph(rng)
            for _ in range(40):
                source, target = rng.sample(list(degrees.people), 2)
                bfs_path = degrees.shortest_path(source, target)
                astar_path = compare.astar(source, target)

                if bfs_path is None:
                    self.assertIsNone(astar_path)
                else:
                    self.assertEqual(len(astar_path), len(bfs_path))
                    self.assertTrue(is_valid_path(source, target, bfs_path))
                    self.assertTrue(is_valid_path(source, target, astar_path))

    def test_same_person(self):
        build_random_graph(random.Random(2))
        self.assertEqual(degrees.shortest_path("p0", "p0"), [])
        self.assertEqual(compare.astar("p0", "p0"), [])


if __name__ == "__main__":
    unittest.main()