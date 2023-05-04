#include <cmath>
#include <iostream>
#include <numbers>

#include "lib.hpp"
#include "rnd.hpp"

using f_t = float;

template <typename T>
struct Params {
	T scaling;
	T decay;
	T frequency;
	T phase;

	template <typename X>
	T evaluate(X&& x) const
	{
		return scaling * std::exp(-x * decay) * std::sin(frequency * x + phase);
	}
};

template <typename T, typename F>
constexpr auto generate_data(T&& start, T&& end, T&& inc, F&& f)
	-> std::pair<std::vector<T>, std::vector<decltype(f(T{}))>>
{
	auto size = static_cast<std::size_t>((end - start) / inc);

	std::vector<T> x;
	std::vector<T> y;
	x.reserve(size);
	y.reserve(size);

	for (std::size_t i = 0; i < size; ++i) {
		auto cur_x = start + static_cast<T>(i) * inc;
		x.push_back(cur_x);
		y.push_back(f(cur_x));
	}

	return { x, y };
}

template <typename T>
std::ostream& operator<<(std::ostream& os, Params<T> const& p)
{
	os << "(scaling=" << p.scaling << ", decay=" << p.decay << ", frequency=" << p.frequency
	   << ", phase=" << p.phase << ")";
	return os;
}

int main()
{
	//auto [min, max, mean, z] = check_uniform_distribution(PCG{}, constexpr_dis<size_t>{0, 10000}, 1000000);
	//std::cout << "min: " << min << " | max: " << max << " | mean : " << mean << " | z: " << z << std::endl;
	static constexpr auto noise_level = 1e-2f;
	static constexpr constexpr_dis<f_t> noise_dis{ 0, noise_level * 2.f };
	constexpr size_t nb_sin = 3;
	constexpr std::array<Params<f_t>, nb_sin> truth = {
		{ { 0.9f, 0.6f, 0.333f, 0.125f }, { 0.5f, 0.1f, 0.6666f, 0.0f }, { 0.4f, 0.01f, 0.4f, 0.f } }
	};
	auto data = generate_data(0.0f, 10.24f, 0.01f, [&](f_t x) {
		PCG pcg{ 0, seed() };
		auto perfect = std::accumulate(truth.cbegin(), truth.cend(), 0.f,
					       [=](auto lhs, auto&& rhs) { return lhs + rhs.evaluate(x); });
		auto noise = noise_dis(pcg) - noise_level;
		return perfect + perfect * noise;
	});
	const auto [x, y] = data;

	static constexpr constexpr_dis<f_t> one_dis{ 0.f, 1.f };
	static constexpr constexpr_dis<size_t> param_dis{ 0, 4 };

	static constexpr auto initializer = [](auto& gen) {
		return Params<f_t>{ one_dis(gen), one_dis(gen), one_dis(gen), one_dis(gen) };
	};
	static constexpr auto mutator = [](auto&& p, auto& gen) {
		auto ret = p;
		switch (param_dis(gen)) {
		case 0:
			ret.scaling = one_dis(gen);
			break;
		case 1:
			ret.decay = one_dis(gen);
			break;
		case 2:
			ret.frequency = one_dis(gen);
			break;
		case 3:
			ret.phase = one_dis(gen);
			break;
		default:
			(void)(0);
		}
		return ret;
	};
	static constexpr auto evaluator = [](auto&& params, auto&& x, auto&& y) {
		auto mse = 0.f;

		for (size_t i = 0; i < x.size(); ++i) {
			const auto xi = x[i];
			const auto yi = y[i];
			auto res = std::accumulate(params.begin(), params.end(), 0.f, [=](auto&& v, auto&& p) {
				return v + (p.scaling * std::exp(-xi * p.decay) * std::sin(p.frequency * xi + p.phase));
			});
			auto diff = yi - res;
			mse += diff * diff;
		}
		return mse;
	};
	static constexpr constexpr_dis<size_t> sin_dis{ 0, nb_sin };
	static constexpr auto optimiser = [](auto& params, auto&& x, auto&& y, auto& gen) {};

	using genome_t = Genome<Params<f_t>, nb_sin, initializer, mutator, evaluator, optimiser>;

	static constexpr auto selector = [](auto&& ind, auto&& evals, auto& gen) -> genome_t& {
		auto dis = constexpr_dis<size_t>{ 0, evals.size() };
		auto idx_first = dis(gen);
		auto idx_second = dis(gen);
		return evals[idx_first] < evals[idx_second] ? ind[idx_first] : ind[idx_second];
	};

	using ea_t = Ea<genome_t, selector>;

	auto ea = ea_t{ 100000 };
	ea.loop(x, y, 250, 0.5);
	decltype(auto) best = ea.best_individual(x, y);
	auto fitness = best.evaluate(x, y);
	decltype(auto) params = best.get_data();

	std::cout << "\n#END#\nBest individual has fitness: " << fitness << "\n";
	for (auto const& p : params) {
		std::cout << p << "\n";
	}
	std::cout << std::endl;

	return 0;
}
