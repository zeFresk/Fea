#ifndef INCLUDE_RND_HPP
#define INCLUDE_RND_HPP

#include <cstdint>
#include <limits>

constexpr auto seed()
{
	std::uint64_t shifted = 0;

	for (const auto c : __TIME__) {
		shifted <<= 8;
		shifted |= c;
	}

	return shifted;
}

struct PCG {
	struct pcg32_random_t {
		std::uint64_t state;
		std::uint64_t inc;
	};
	pcg32_random_t rng;
	typedef std::uint32_t result_type;

	constexpr result_type operator()()
	{
		return pcg32_random_r();
	}

	static constexpr result_type min()
	{
		return std::numeric_limits<result_type>::min();
	}

	static constexpr result_type max()
	{
		return std::numeric_limits<result_type>::max();
	}

    private:
	constexpr std::uint32_t pcg32_random_r()
	{
		std::uint64_t oldstate = rng.state;
		// Advance internal state
		rng.state = oldstate * 6364136223846793005ULL + (rng.inc | 1);
		// Calculate output function (XSH RR), uses old state for max ILP
		std::uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
		std::uint32_t rot = oldstate >> 59u;
		return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
	}
};

template <typename T>
struct constexpr_dis {
	T min;
	T max;

	template <typename Gen>
	constexpr T operator()(Gen&& gen) const
	{
		// https://math.stackexchange.com/questions/314244/scaling-a-uniform-distribution-probability
		const auto span = max - min;
		const auto v = static_cast<T>(gen());
		const auto moved = v - static_cast<T>(gen.min()); // [0, b-a]
		const auto scaled = (moved * span) / (static_cast<T>(gen.max()) - static_cast<T>(gen.min())); // [0, MAX - MIN]
		return scaled+min; // [MIN, MAX]
	}
};

#include <cmath>
#include <tuple>

template <typename Gen, typename Dis>
std::tuple<float, float, float, float> check_uniform_distribution(Gen&& gen, Dis&& dis, std::size_t N) {
	float span = static_cast<float>(dis.max - dis.min);
	float expected_mean = span/2.f;
	float expected_var = span * span / 12.f;

	float cur_sum = 0.f;
	float cur_min = dis.max;
	float cur_max = dis.min;
	for (std::size_t i = 0; i < N; ++i) {
		auto r = static_cast<float>(dis(gen));
		cur_sum += r;
		cur_min = std::min(cur_min, r);
		cur_max = std::max(cur_max, r);
	}
	float emean = cur_sum / N;
	float z = std::sqrt(N) * (emean - expected_mean) / (std::sqrt(expected_var));
	return {cur_min, cur_max, emean, z};
}

#endif
