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
		const auto v = static_cast<T>(gen());
		const auto moved = v - static_cast<T>(gen.min());
		const auto scaled = (moved * (max-min)) / (static_cast<T>(gen.max()) - static_cast<T>(gen.min()));
		return scaled-min;
	}
};

#endif
