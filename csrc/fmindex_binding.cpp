// BEguidedOFF fast-mode FM-index binding.
//
// Wraps an sdsl-lite compressed suffix array (`csa_wt`) over one concatenated,
// PAM-agnostic genome.  Chromosomes are concatenated with a separator symbol so
// that exact anchors can never span a chromosome boundary; a sidecar `.meta`
// file records the chromosome name / length / offset table used to map global
// positions back to (chromosome, local position).
//
// Symbol alphabet (byte values):
//   1 = A, 2 = C, 3 = G, 4 = T, 6 = separator (never matched)
// The Python codec maps every non-ACGT byte (incl. N) to 1 = A, mirroring the
// 2-bit encoding of the extensive mode, so symbol 5 never occurs in practice.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <sdsl/suffix_arrays.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace py = pybind11;
using csa_t = sdsl::csa_wt<>;

namespace {

constexpr uint8_t SEPARATOR = 6;

struct ChromTable {
    std::vector<std::string> names;
    std::vector<uint64_t> lengths;
    std::vector<uint64_t> offsets;
};

std::string meta_path_for(const std::string& index_path) {
    return index_path + ".meta";
}

void write_meta(const std::string& index_path, const ChromTable& chroms) {
    std::ofstream out(meta_path_for(index_path));
    if (!out) {
        throw std::runtime_error("cannot write meta file: " + meta_path_for(index_path));
    }
    out << "#beguidedoff-fmindex\t1\n";
    for (size_t i = 0; i < chroms.names.size(); ++i) {
        out << chroms.names[i] << '\t' << chroms.lengths[i] << '\t' << chroms.offsets[i] << '\n';
    }
}

ChromTable read_meta(const std::string& index_path) {
    std::ifstream in(meta_path_for(index_path));
    if (!in) {
        throw std::runtime_error("cannot read meta file: " + meta_path_for(index_path));
    }
    ChromTable chroms;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string name, len_s, off_s;
        std::getline(ss, name, '\t');
        std::getline(ss, len_s, '\t');
        std::getline(ss, off_s, '\t');
        chroms.names.push_back(name);
        chroms.lengths.push_back(std::stoull(len_s));
        chroms.offsets.push_back(std::stoull(off_s));
    }
    if (chroms.names.empty()) {
        throw std::runtime_error("empty chromosome table in " + meta_path_for(index_path));
    }
    return chroms;
}

}  // namespace

class FMIndex {
public:
    FMIndex() = default;

    static FMIndex load(const std::string& index_path) {
        FMIndex idx;
        sdsl::load_from_file(idx.csa_, index_path);
        idx.chroms_ = read_meta(index_path);
        return idx;
    }

    static void build(const std::string& text_path,
                      const std::string& out_path,
                      const std::vector<std::string>& names,
                      const std::vector<int64_t>& offsets,
                      const std::vector<int64_t>& lengths) {
        if (names.size() != offsets.size() || names.size() != lengths.size()) {
            throw std::runtime_error("chromosome name/offset/length lists differ in length");
        }
        csa_t csa;
        // sdsl's construct() writes intermediate files (suffix array, BWT, ...)
        // named after the cache id inside the cache dir.  A fixed id in the
        // current working directory makes concurrent builds clobber each
        // other's files, so every build call gets its own private directory
        // next to the output, removed again afterwards.
        std::filesystem::path out_dir = std::filesystem::path(out_path).parent_path();
        if (out_dir.empty()) out_dir = ".";
        const std::string cache_id =
            "beguidedoff_cache_" + std::to_string(::getpid()) + "_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        const std::string cache_dir = (out_dir / cache_id).string();
        std::filesystem::create_directories(cache_dir);
        try {
            sdsl::cache_config config(true, cache_dir, "c_");
            // num_bytes = 1: text_path is a raw sequence of byte symbols (1..6).
            sdsl::construct(csa, text_path, config, 1);

            sdsl::store_to_file(csa, out_path);

            ChromTable chroms;
            chroms.names = names;
            chroms.offsets.assign(offsets.begin(), offsets.end());
            chroms.lengths.assign(lengths.begin(), lengths.end());
            write_meta(out_path, chroms);
        } catch (...) {
            std::error_code ec;
            std::filesystem::remove_all(cache_dir, ec);
            throw;
        }
        std::error_code ec;
        std::filesystem::remove_all(cache_dir, ec);
    }

    uint64_t size() const { return csa_.size(); }

    // Returns the SA interval [l, r] for an exact anchor, or an empty interval
    // encoded as (1, 0) when the anchor does not occur.
    std::pair<uint64_t, uint64_t> interval(const std::string& anchor) const {
        if (anchor.empty()) return {1, 0};
        uint64_t l = 0, r = 0;
        auto n = sdsl::backward_search(
            csa_, 0, csa_.size() - 1,
            anchor.begin(), anchor.end(), l, r);
        if (n == 0) return {1, 0};
        return {l, r};
    }

    uint64_t count(const std::string& anchor) const {
        auto iv = interval(anchor);
        if (iv.second < iv.first) return 0;
        return iv.second - iv.first + 1;
    }

    // Locate up to `max_hits` occurrences of an exact anchor.  `max_hits <= 0`
    // means "all".  Returns global 0-based offsets into the concatenated text.
    py::array_t<uint64_t> locate(const std::string& anchor, int64_t max_hits) const {
        auto iv = interval(anchor);
        return locate_interval(iv.first, iv.second, max_hits);
    }

    py::array_t<uint64_t> locate_interval(uint64_t l, uint64_t r, int64_t max_hits) const {
        std::vector<uint64_t> out;
        if (r < l) {
            return to_array(out);
        }
        const uint64_t n = r - l + 1;
        const uint64_t limit = (max_hits <= 0) ? n : std::min<uint64_t>(n, static_cast<uint64_t>(max_hits));
        out.reserve(limit);
        for (uint64_t i = 0; i < limit; ++i) {
            out.push_back(csa_[l + i]);
        }
        return to_array(out);
    }

    // Extract `length` symbols starting at global offset `pos`.
    py::bytes extract(uint64_t pos, uint64_t length) const {
        if (length == 0) return py::bytes("", 0);
        std::string buf(length, '\0');
        sdsl::extract(csa_, pos, pos + length - 1, buf.begin());
        return py::bytes(buf);
    }

    std::vector<std::string> chrom_names() const { return chroms_.names; }
    std::vector<int64_t> chrom_lengths() const {
        return std::vector<int64_t>(chroms_.lengths.begin(), chroms_.lengths.end());
    }
    std::vector<int64_t> chrom_offsets() const {
        return std::vector<int64_t>(chroms_.offsets.begin(), chroms_.offsets.end());
    }

private:
    static py::array_t<uint64_t> to_array(const std::vector<uint64_t>& v) {
        py::array_t<uint64_t> arr(v.size());
        if (!v.empty()) {
            std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(uint64_t));
        }
        return arr;
    }

    csa_t csa_;
    ChromTable chroms_;
};

PYBIND11_MODULE(_fmindex, m) {
    m.doc() = "sdsl-lite FM-index binding for BEguidedOFF fast mode";

    py::class_<FMIndex>(m, "FMIndex")
        .def_static("load", &FMIndex::load, py::arg("index_path"))
        .def_static("build", &FMIndex::build,
                    py::arg("text_path"), py::arg("out_path"),
                    py::arg("names"), py::arg("offsets"), py::arg("lengths"))
        .def("size", &FMIndex::size)
        .def("interval", &FMIndex::interval, py::arg("anchor"))
        .def("count", &FMIndex::count, py::arg("anchor"))
        .def("locate", &FMIndex::locate, py::arg("anchor"), py::arg("max_hits") = 0)
        .def("locate_interval", &FMIndex::locate_interval,
             py::arg("l"), py::arg("r"), py::arg("max_hits") = 0)
        .def("extract", &FMIndex::extract, py::arg("pos"), py::arg("length"))
        .def("chrom_names", &FMIndex::chrom_names)
        .def("chrom_lengths", &FMIndex::chrom_lengths)
        .def("chrom_offsets", &FMIndex::chrom_offsets);

    m.attr("SEPARATOR") = static_cast<int>(SEPARATOR);
}
