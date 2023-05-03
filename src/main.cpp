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

int main()
{
	constexpr auto noise_level = 1e-2f;
	constexpr size_t nb_sin = 1;
	constexpr std::array<Params<f_t>, nb_sin> truth = { { { 1.4f, 1.421f, 0.333f, 0.125f } } };
	auto data = generate_data(0.0f, 10.24f, 0.01f, [&](f_t x) {
		return std::accumulate(truth.cbegin(), truth.cend(), 0.f,
				       [=](auto lhs, auto&& rhs) { return lhs + rhs.evaluate(x); });
	});
	const auto [x, y] = data;

	static constexpr constexpr_dis<f_t> dpi_dis{ 0.f, 2.f * std::numbers::pi_v<f_t> };
	static constexpr constexpr_dis<f_t> scale_dis{ 0.f, 10.f };
	static constexpr constexpr_dis<f_t> decay_dis{ 0.f, 1.0f };
	static constexpr constexpr_dis<size_t> param_dis{ 1, 4 };

	static constexpr auto initializer = [](auto& gen) {
		return Params<f_t>{ scale_dis(gen), decay_dis(gen), dpi_dis(gen), dpi_dis(gen) };
	};
	static constexpr auto mutator = [](auto&& p, auto& gen) {
		auto ret = p;
		switch (param_dis(gen)) {
		case 1:
			ret.scaling = scale_dis(gen);
			break;
		case 2:
			ret.decay = decay_dis(gen);
			break;
		case 3:
			ret.frequency = dpi_dis(gen);
			break;
		case 4:
			ret.phase = dpi_dis(gen);
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

	using genome_t = Genome<Params<f_t>, nb_sin, initializer, mutator, evaluator>;

	static constexpr auto selector = [](auto&& ind, auto&& evals, auto& gen) -> genome_t& {
		auto dis = constexpr_dis<size_t>{ 0, evals.size() };
		auto idx_first = dis(gen);
		auto idx_second = dis(gen);
		return evals[idx_first] < evals[idx_second] ? ind[idx_first] : ind[idx_second];
	};

	using ea_t = Ea<genome_t, selector>;

	auto ea = ea_t{ 1000 };
	ea.loop(x, y, 100, 0.1);

	return 0;
}
