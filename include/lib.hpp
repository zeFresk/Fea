#ifndef INCLUDE_LIB_HPP
#define INCLUDE_LIB_HPP

#include <array>
#include <chrono>
#include <execution>
#include <iostream>
#include <optional>
#include <random>
#include <vector>
#include <type_traits>

#include "rnd.hpp"

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
struct return_type : return_type<decltype(&T::operator())> {
};

template <typename ClassType, typename ReturnType, typename... Args>
struct return_type<ReturnType (ClassType::*)(Args...) const> {
	using type = ReturnType;
};

template <typename P, std::size_t N, auto initializer, auto mutator, auto evaluator>
class Genome
{
	//using helper_t = typename decltype(std::function{ evaluator })::result_type;
	std::array<P, N> data;
	//std::optional<helper_t> evaluation;

    public:
	constexpr Genome() : data{} //, evaluation{ std::nullopt }
	{
	}
	constexpr Genome(Genome const&) = delete;
	constexpr Genome(Genome&&) noexcept = default;
	constexpr ~Genome() = default;
	constexpr Genome& operator=(Genome const&) = delete;
	constexpr Genome& operator=(Genome&&) noexcept = default;

	template <typename Gen>
	constexpr void random_reset(Gen& gen)
	{
		std::generate(data.begin(), data.end(), [&]() { return initializer(gen); });
		//if (evaluation)
		//		evaluation = std::nullopt;
	}

	template <typename Gen>
	constexpr Genome xover(Genome const& oth, Gen& gen) const
	{
		Genome nouveau{};

		size_t loc;
		if consteval {
			auto dis = constexpr_dis<size_t>{ 0, N - 1 };
			loc = dis(gen);
		} else {
			auto dis = std::uniform_int_distribution<>(0, N - 1);
			loc = dis(gen);
		}
		std::copy(data.cbegin(), data.cbegin() + loc, nouveau.data.begin());
		std::copy(oth.data.cbegin(), oth.data.cbegin() + (N - loc), nouveau.data.begin() + loc);
		return nouveau;
	}

	template <typename Gen>
	constexpr void mutate(float p_mutation, Gen& gen)
	{
		auto dis = constexpr_dis<float>{ 0.f, 1.f };
		auto p = dis(gen);
		if (p <= p_mutation)
			mutate_once(gen);
	}

	template <typename Gen>
	constexpr void mutate_once(Gen& gen)
	{
		size_t i = 0;
		if consteval {
			auto dis = constexpr_dis<size_t>{ 0, N - 1 };
			i = dis(gen);
		} else {
			auto dis = std::uniform_int_distribution<>(0, N - 1);
			i = dis(gen);
		}
		data[i] = mutator(data[i], gen);
		//if (evaluation)
		//	evaluation = std::nullopt;
	}

	template <typename X, typename Y>
	constexpr auto evaluate(X&& x, Y&& y) const
	{
		//if (!evaluation) {
		//	evaluation = evaluator(data, std::forward<X>(x), std::forward<Y>(y));
		//}
		return evaluator(data, std::forward<X>(x), std::forward<Y>(y)); //*evaluation;
	}

	constexpr decltype(data) const& get_data() const
	{
		return data;
	}

	//using eval_t = helper_t;
};

template <typename Genome, auto selector>
class Ea
{
	std::vector<Genome> individuals;
	std::vector<Genome> childrens;
	PCG gen;

    public:
	constexpr Ea(size_t n) : individuals(n), childrens(n), gen{ 0, seed() }
	{
		if consteval {
			std::for_each(individuals.begin(), individuals.end(),
				      [this](auto&& ind) { ind.random_reset(this->gen); });
		} else {
			std::for_each(std::execution::par, individuals.begin(), individuals.end(),
				      [this](auto&& ind) { ind.random_reset(this->gen); });
		}
	}

	template <typename X, typename Y>
	constexpr void loop(X&& x, Y&& y, std::size_t nb_gens, float p_mutation)
	{
		if consteval {
			constexpr_loop(std::forward<X>(x), std::forward<Y>(y), nb_gens, p_mutation);
		} else {
			runtime_loop(std::forward<X>(x), std::forward<Y>(y), nb_gens, p_mutation);
		}
	}

	template <typename X, typename Y>
	constexpr Genome const& best_individual(X&& x, Y&& y) const
	{
		std::vector<decltype(std::declval<Genome>().evaluate(x, y))> evaluations(individuals.size());
		auto best_it = evaluations.cend();
		if consteval {
			std::transform(individuals.cbegin(), individuals.cend(), evaluations.begin(),
				       [&](auto&& genome) { return genome.evaluate(x, y); });
			best_it = std::min_element(evaluations.cbegin(), evaluations.cend());
		} else {
			std::transform(std::execution::par, individuals.cbegin(), individuals.cend(),
				       evaluations.begin(), [&](auto&& genome) { return genome.evaluate(x, y); });
			best_it = std::min_element(std::execution::par, evaluations.cbegin(), evaluations.cend());
		}
		auto best_idx = std::distance(evaluations.cbegin(), best_it);
		return individuals[best_idx];
	}

    private:
	template <typename X, typename Y>
	constexpr void constexpr_loop(X&& x, Y&& y, std::size_t nb_gens, float p_mutation)
	{
		std::vector<decltype(std::declval<Genome>().evaluate(x, y))> evaluations(individuals.size());
		for (size_t i = 0; i < nb_gens; ++i) {
			std::transform(individuals.cbegin(), individuals.cend(), evaluations.begin(),
				       [&](auto&& genome) { return genome.evaluate(x, y); });
			std::for_each(childrens.begin(), childrens.end(), [&, this](auto& child) {
				auto& parent1 = selector(individuals, evaluations, this->gen);
				auto& parent2 = selector(individuals, evaluations, this->gen);
				auto ret = parent1.xover(parent2, this->gen);
				ret.mutate(p_mutation, this->gen);
				child = std::move(ret);
			});
			std::swap(childrens, individuals);
		}
	}

	template <typename X, typename Y>
	void runtime_loop(X&& x, Y&& y, std::size_t nb_gens, float p_mutation)
	{
		std::vector<decltype(std::declval<Genome>().evaluate(x, y))> evaluations(individuals.size());
		auto started_at = std::chrono::system_clock::now();
		for (size_t i = 0; i < nb_gens; ++i) {
			std::transform(std::execution::par, individuals.cbegin(), individuals.cend(),
				       evaluations.begin(), [&](auto&& genome) { return genome.evaluate(x, y); });
			std::generate(std::execution::par, childrens.begin(), childrens.end(), [&, this]() {
				auto& parent1 = selector(individuals, evaluations, this->gen);
				auto& parent2 = selector(individuals, evaluations, this->gen);
				auto ret = parent1.xover(parent2, this->gen);
				ret.mutate(p_mutation, this->gen);
				return std::move(ret);
			});
			auto best_it = std::min_element(std::execution::par, evaluations.cbegin(), evaluations.cend());
			auto best_idx = std::distance(evaluations.cbegin(), best_it);
			childrens[0] = std::move(individuals[best_idx]);
			std::swap(childrens, individuals);

			auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::system_clock::now() - started_at);
			std::cout << "Gen " << i + 1 << " : best individual has fitness = "
				  << *std::min_element(std::execution::par, evaluations.cbegin(), evaluations.cend())
				  << " [" << elapsed.count() / 1000.f << "s]" << std::endl;
		}
	}
};

#endif
