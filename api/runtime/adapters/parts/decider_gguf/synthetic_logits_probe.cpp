#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>
#include "sha256.h"

namespace {
using json = nlohmann::json;

struct ImageRef {
    std::string path, artifact_id, digest, kind;
};
struct Request {
    std::string part_id, topology_revision, prompt, prompt_digest;
    uint32_t view_index = 0;
    ImageRef image;
};

static std::string file_sha256(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file: " + path);
    sha256_t state;
    sha256_init(&state);
    std::vector<unsigned char> chunk(1024 * 1024);
    while (in) {
        in.read(reinterpret_cast<char *>(chunk.data()), static_cast<std::streamsize>(chunk.size()));
        const auto got = in.gcount();
        if (got > 0) sha256_update(&state, chunk.data(), static_cast<size_t>(got));
    }
    if (!in.eof()) throw std::runtime_error("read failure: " + path);
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&state, digest);
    static const char hex[] = "0123456789abcdef";
    std::string result = "sha256:";
    for (unsigned char byte : digest) {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}

static std::string required_string(const json & object, const char * key) {
    if (!object.is_object() || !object.contains(key) || !object.at(key).is_string())
        throw std::runtime_error(std::string("missing/non-string field: ") + key);
    std::string value = object.at(key).get<std::string>();
    if (value.empty()) throw std::runtime_error(std::string("empty field: ") + key);
    return value;
}

static bool valid_sha256(const std::string & digest) {
    if (digest.size() != 71 || digest.compare(0, 7, "sha256:") != 0) return false;
    return std::all_of(digest.begin() + 7, digest.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

static std::vector<Request> read_requests(const std::string & filename) {
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("cannot open JSONL requests: " + filename);
    std::vector<Request> requests;
    std::unordered_set<std::string> completed_parts, artifact_ids;
    std::string current_part, current_topology, current_prompt;
    uint32_t expected_view = 0;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) throw std::runtime_error("blank JSONL row at line " + std::to_string(line_number));
        json row;
        try { row = json::parse(line); }
        catch (const std::exception & e) {
            throw std::runtime_error("malformed JSON at line " + std::to_string(line_number) + ": " + e.what());
        }
        Request request;
        request.part_id = required_string(row, "part_id");
        request.topology_revision = required_string(row, "topology_revision");
        if (!valid_sha256(request.topology_revision))
            throw std::runtime_error("topology_revision must be a sha256 digest at line " + std::to_string(line_number));
        request.prompt = required_string(row, "prompt");
        if (!row.contains("view_index") || !row.at("view_index").is_number_integer())
            throw std::runtime_error("view_index must be an integer at line " + std::to_string(line_number));
        const int64_t view_index = row.at("view_index").get<int64_t>();
        if (view_index < 0 || view_index > 3)
            throw std::runtime_error("view_index must be an integer in [0,3] at line " + std::to_string(line_number));
        request.view_index = static_cast<uint32_t>(view_index);
        if (request.prompt.size() < 9 || request.prompt.compare(request.prompt.size() - 9, 9, "Answer: (") != 0)
            throw std::runtime_error("prompt must end exactly with Decider answer slot 'Answer: (' at line " + std::to_string(line_number));
        sha256_t prompt_hash;
        sha256_init(&prompt_hash);
        sha256_update(&prompt_hash, reinterpret_cast<const unsigned char *>(request.prompt.data()), request.prompt.size());
        unsigned char prompt_bytes[SHA256_DIGEST_SIZE];
        sha256_final(&prompt_hash, prompt_bytes);
        static const char hex[] = "0123456789abcdef";
        request.prompt_digest = "sha256:";
        for (unsigned char byte : prompt_bytes) { request.prompt_digest += hex[byte >> 4]; request.prompt_digest += hex[byte & 15]; }

        if (!current_part.empty() && request.part_id != current_part) {
            if (expected_view != 4)
                throw std::runtime_error("each part must contain exactly ordered view_index values 0,1,2,3");
            completed_parts.insert(current_part);
            if (completed_parts.count(request.part_id))
                throw std::runtime_error("rows for a part must be contiguous");
            expected_view = 0;
        }
        if (current_part.empty() || request.part_id != current_part) {
            current_part = request.part_id;
            current_topology = request.topology_revision;
            current_prompt = request.prompt;
        } else if (request.topology_revision != current_topology) {
            throw std::runtime_error("all views for a part must use one topology_revision");
        }
        if (request.prompt != current_prompt)
            throw std::runtime_error("all per-view requests must use identical frozen prompt bytes");
        if (request.view_index != expected_view)
            throw std::runtime_error("each part must contain exactly ordered view_index values 0,1,2,3");
        ++expected_view;
        if (!row.contains("image") || !row.at("image").is_object() || row.at("image").size() != 4)
            throw std::runtime_error("each row must contain exactly one image object with path/artifact_id/digest/kind at line " + std::to_string(line_number));
        const auto & item = row.at("image");
        for (const char * field : {"path", "artifact_id", "digest", "kind"})
            if (!item.contains(field)) throw std::runtime_error(std::string("image reference missing ") + field + " at line " + std::to_string(line_number));
        request.image = ImageRef { required_string(item, "path"), required_string(item, "artifact_id"),
                                   required_string(item, "digest"), required_string(item, "kind") };
        if (!valid_sha256(request.image.digest))
            throw std::runtime_error("invalid image SHA-256 syntax at line " + std::to_string(line_number));
        if (!artifact_ids.insert(request.part_id + "\n" + request.image.artifact_id).second)
            throw std::runtime_error("duplicate artifact_id within part at line " + std::to_string(line_number));
        if (file_sha256(request.image.path) != request.image.digest)
            throw std::runtime_error("image digest mismatch at line " + std::to_string(line_number) + ": " + request.image.path);
        size_t markers = 0, position = 0;
        const std::string marker = mtmd_default_marker();
        while ((position = request.prompt.find(marker, position)) != std::string::npos) { ++markers; position += marker.size(); }
        if (markers > 1)
            throw std::runtime_error("one-image prompt may contain at most one media marker at line " + std::to_string(line_number));
        requests.push_back(std::move(request));
    }
    if (!input.eof()) throw std::runtime_error("failed reading JSONL request file");
    if (requests.empty()) throw std::runtime_error("JSONL request file has no rows");
    if (expected_view != 4)
        throw std::runtime_error("each part must end with exactly ordered view_index values 0,1,2,3");
    for (const auto & request : requests) {
        if (request.prompt_digest != requests.front().prompt_digest)
            throw std::runtime_error("all batch rows must use the same frozen prompt bytes");
    }
    return requests;
}

static std::string model_digest(const std::string & path) { return file_sha256(path); }

static int run_batch(const std::string & model_path, const std::string & mmproj_path,
                     const std::string & requests_path) {
    std::vector<Request> requests = read_requests(requests_path);
    const std::string text_digest = model_digest(model_path);
    const std::string projector_digest = model_digest(mmproj_path);
    ggml_backend_load_all();
    ggml_backend_reg_t hip = ggml_backend_reg_by_name("ROCm");
    if (!hip) throw std::runtime_error("ROCm/HIP backend registry is not loaded; refusing generic GPU fallback");
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(hip); ++i) {
        ggml_backend_dev_t candidate = ggml_backend_reg_dev_get(hip, i);
        if (candidate && ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = candidate;
            break;
        }
    }
    if (!gpu) throw std::runtime_error("no GPU backend device found");
    const std::string registry_name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(gpu));
    if (registry_name != "ROCm") throw std::runtime_error("selected GPU is not owned by the ROCm/HIP backend registry");
    const std::string backend_name = "HIP";
    const std::string device_id = ggml_backend_dev_name(gpu);
    const std::string device_name = ggml_backend_dev_description(gpu);
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    model_params.check_tensors = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) throw std::runtime_error("text model load failed");
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 4096; context_params.n_batch = 2048;
    context_params.n_ubatch = 512; context_params.n_threads = 8;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) { llama_model_free(model); llama_backend_free(); throw std::runtime_error("context init failed"); }
    auto vision_params = mtmd_context_params_default();
    vision_params.use_gpu = true; vision_params.device = gpu;
    vision_params.n_threads = 8; vision_params.print_timings = false;
    mtmd_context * vision = mtmd_init_from_file(mmproj_path.c_str(), model, vision_params);
    if (!vision || !mtmd_support_vision(vision)) {
        if (vision) mtmd_free(vision);
        llama_free(context); llama_model_free(model); llama_backend_free();
        throw std::runtime_error("multimodal projector load/vision support failed");
    }
    // Decode every image before writing stdout. This makes codec/path failures
    // fail closed without emitting a partial result stream.
    for (const auto & request : requests) {
        mtmd_helper_bitmap_wrapper decoded = mtmd_helper_bitmap_init_from_file(
            vision, request.image.path.c_str(), false, mtmd_helper_init_opt_default());
        if (!decoded.bitmap) {
            mtmd_free(vision); llama_free(context); llama_model_free(model); llama_backend_free();
            throw std::runtime_error("pinned mtmd image decoder rejected input: " + request.image.path);
        }
        mtmd_bitmap_free(decoded.bitmap);
        if (decoded.video_ctx) mtmd_helper_video_free(decoded.video_ctx);
    }
    const std::vector<std::string> options = {"A","B","C","D","E","F","G","H","I","J"};
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const float * unused_logits = nullptr;
    (void) unused_logits;
    std::vector<llama_token> token_ids;
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    for (const auto & option : options) {
        llama_token token = LLAMA_TOKEN_NULL;
        const int32_t n = llama_tokenize(vocab, option.c_str(), option.size(), &token, 1, false, false);
        if (n != 1 || token < 0 || token >= vocab_size ||
            std::find(token_ids.begin(), token_ids.end(), token) != token_ids.end()) {
            mtmd_free(vision); llama_free(context); llama_model_free(model); llama_backend_free();
            throw std::runtime_error("answer choice token is invalid or nonunique: " + option);
        }
        token_ids.push_back(token);
    }
    std::cout << json{{"schema","modly.decider.slot-scores.v2"},{"record","header"},
                      {"backend",backend_name},{"device",device_name},
                      {"device_id",device_id},
                      {"backend_registry",registry_name},
                      {"runtime","ROCm 7.14.60850"},
                      {"build_id","llama.cpp@9575389609d6f8437de0b205561a4824d217c409;HIP;gfx1100;ROCm-7.14.60850;mtmd-jsonl-per-view-v2"},
                      {"aggregation","equal_weight_mean_of_four_view_probabilities"},
                      {"prompt_digest",requests.front().prompt_digest},
                      {"model_digest",text_digest},{"mmproj_digest",projector_digest}}.dump() << std::endl;
    for (const Request & request : requests) {
        llama_memory_clear(llama_get_memory(context), true);
        std::vector<mtmd_bitmap *> owned;
        std::vector<const mtmd_bitmap *> bitmaps;
        try {
            if (file_sha256(request.image.path) != request.image.digest)
                throw std::runtime_error("image changed after request validation: " + request.image.path);
            mtmd_helper_bitmap_wrapper decoded = mtmd_helper_bitmap_init_from_file(
                vision, request.image.path.c_str(), false, mtmd_helper_init_opt_default());
            if (!decoded.bitmap) throw std::runtime_error("pinned mtmd image decoder rejected input: " + request.image.path);
            owned.push_back(decoded.bitmap); bitmaps.push_back(decoded.bitmap);
            if (decoded.video_ctx) mtmd_helper_video_free(decoded.video_ctx);
            std::string prompt = request.prompt;
            const std::string marker = mtmd_default_marker();
            size_t markers = 0, position = 0;
            while ((position = prompt.find(marker, position)) != std::string::npos) { ++markers; position += marker.size(); }
            if (markers == 0) prompt.insert(0, marker + "\n");
            mtmd_input_text text { prompt.data(), prompt.size(), true, true };
            mtmd_input_chunks * chunks = mtmd_input_chunks_init();
            const int32_t tokenized = mtmd_tokenize(vision, chunks, &text, bitmaps.data(), bitmaps.size());
            if (tokenized != 0) { mtmd_input_chunks_free(chunks); throw std::runtime_error("multimodal tokenize failed: " + std::to_string(tokenized)); }
            llama_pos n_past = 0;
            const int32_t evaluated = mtmd_helper_eval_chunks(vision, context, chunks, n_past, 0, context_params.n_batch, true, &n_past);
            mtmd_input_chunks_free(chunks);
            if (evaluated != 0) throw std::runtime_error("multimodal prompt evaluation failed: " + std::to_string(evaluated));
            const float * logits = llama_get_logits_ith(context, -1);
            if (!logits) throw std::runtime_error("answer-slot logits unavailable");
            std::vector<float> values;
            for (llama_token token : token_ids) values.push_back(logits[token]);
            const float max_logit = *std::max_element(values.begin(), values.end());
            std::vector<double> exponents;
            for (float value : values) exponents.push_back(std::exp(static_cast<double>(value - max_logit)));
            const double denominator = std::accumulate(exponents.begin(), exponents.end(), 0.0);
            json evidence{{"artifact_id",request.image.artifact_id},
                          {"digest",request.image.digest},{"kind",request.image.kind}};
            json scored_options = json::array();
            for (size_t i = 0; i < options.size(); ++i)
                scored_options.push_back(json{{"letter",options[i]},{"token_id",token_ids[i]},
                                               {"logit",values[i]},{"probability",exponents[i] / denominator}});
            json record{{"schema","modly.decider.slot-scores.v2"},{"record","scores"},
                        {"part_id",request.part_id},{"topology_revision",request.topology_revision},
                        {"view_index",request.view_index},{"view_id",request.image.artifact_id},
                        {"evidence",evidence},{"options",scored_options},
                        {"score_kind","native_logits_at_answer_open_paren"},
                        {"confidence","uncalibrated"},{"prompt_digest",request.prompt_digest},
                        {"prompt_tokens_including_media",n_past}};
            std::cout << record.dump() << std::endl;
            for (auto * bitmap : owned) mtmd_bitmap_free(bitmap);
        } catch (...) {
            for (auto * bitmap : owned) mtmd_bitmap_free(bitmap);
            mtmd_free(vision); llama_free(context); llama_model_free(model); llama_backend_free();
            throw;
        }
    }
    mtmd_free(vision); llama_free(context); llama_model_free(model); llama_backend_free();
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "llama.cpp@9575389609d6f8437de0b205561a4824d217c409;HIP;gfx1100;ROCm-7.14.60850;mtmd-jsonl-per-view-v2\n";
        return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "--model" && std::string(argv[3]) == "--mmproj") {
        std::cerr << "usage: probe --model TEXT --mmproj MMPROJ --input-jsonl REQUESTS\n";
        return 2;
    }
    if (argc == 6 && std::string(argv[1]) == "--model" && std::string(argv[3]) == "--mmproj" &&
        std::string(argv[5]) == "--input-jsonl") {
        std::cerr << "usage: probe --model TEXT --mmproj MMPROJ --input-jsonl REQUESTS\n";
        return 2;
    }
    if (argc == 7 && std::string(argv[1]) == "--model" && std::string(argv[3]) == "--mmproj" &&
        std::string(argv[5]) == "--input-jsonl") {
        try { return run_batch(argv[2], argv[4], argv[6]); }
        catch (const std::exception & e) { std::cerr << "batch probe failed: " << e.what() << "\n"; return 2; }
    }
    if (argc < 7) {
        std::cerr << "usage: probe TEXT_GGUF MMPROJ_GGUF WIDTH HEIGHT PROMPT_FILE RGB_FILE [RGB_FILE ...]\n";
        return 2;
    }
    const std::string model_path = argv[1], mmproj_path = argv[2];
    const uint32_t width = static_cast<uint32_t>(std::stoul(argv[3]));
    const uint32_t height = static_cast<uint32_t>(std::stoul(argv[4]));
    std::ifstream prompt_file(argv[5], std::ios::binary);
    if (!prompt_file) { std::cerr << "prompt open failed\n"; return 2; }
    std::string prompt((std::istreambuf_iterator<char>(prompt_file)), {});

    std::vector<mtmd_bitmap *> owned_bitmaps;
    std::vector<const mtmd_bitmap *> bitmaps;
    for (int i = 6; i < argc; ++i) {
        std::ifstream file(argv[i], std::ios::binary);
        if (!file) { std::cerr << "RGB image open failed: " << argv[i] << "\n"; return 2; }
        std::vector<unsigned char> rgb((std::istreambuf_iterator<char>(file)), {});
        if (rgb.size() != static_cast<size_t>(width) * height * 3) {
            std::cerr << "RGB byte count does not match dimensions: " << argv[i] << "\n"; return 2;
        }
        mtmd_bitmap * bitmap = mtmd_bitmap_init(width, height, rgb.data());
        if (!bitmap) { std::cerr << "image bitmap init failed\n"; return 2; }
        owned_bitmaps.push_back(bitmap);
        bitmaps.push_back(bitmap);
    }
    const std::string marker = mtmd_default_marker();
    size_t marker_count = 0, marker_pos = 0;
    while ((marker_pos = prompt.find(marker, marker_pos)) != std::string::npos) {
        ++marker_count; marker_pos += marker.size();
    }
    if (marker_count == 0) {
        for (size_t i = 0; i < bitmaps.size(); ++i) prompt.insert(0, marker + "\n");
    } else if (marker_count != bitmaps.size()) {
        std::cerr << "media-marker count does not equal input image count\n"; return 2;
    }

    ggml_backend_load_all();
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu) { std::cerr << "no GPU backend device found\n"; return 3; }
    const char * device_name = ggml_backend_dev_name(gpu);
    const char * device_desc = ggml_backend_dev_description(gpu);
    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    model_params.check_tensors = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) { std::cerr << "text model load failed\n"; return 4; }
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 4096;
    context_params.n_batch = 2048;
    context_params.n_ubatch = 512;
    context_params.n_threads = 8;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) { std::cerr << "context init failed\n"; llama_model_free(model); return 5; }
    auto vision_params = mtmd_context_params_default();
    vision_params.use_gpu = true;
    vision_params.device = gpu;
    vision_params.n_threads = 8;
    vision_params.print_timings = true;
    mtmd_context * vision = mtmd_init_from_file(mmproj_path.c_str(), model, vision_params);
    if (!vision || !mtmd_support_vision(vision)) {
        std::cerr << "multimodal projector load/vision support failed\n";
        if (vision) mtmd_free(vision);
        llama_free(context); llama_model_free(model); llama_backend_free();
        return 6;
    }

    mtmd_input_text text { prompt.data(), prompt.size(), true, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const int32_t tokenized = mtmd_tokenize(vision, chunks, &text, bitmaps.data(), bitmaps.size());
    if (tokenized != 0) {
        std::cerr << "multimodal tokenize failed: " << tokenized << "\n";
        mtmd_input_chunks_free(chunks);
        for (auto * b : owned_bitmaps) mtmd_bitmap_free(b);
        mtmd_free(vision); llama_free(context); llama_model_free(model); llama_backend_free(); return 7;
    }
    llama_pos n_past = 0;
    const int32_t eval = mtmd_helper_eval_chunks(vision, context, chunks, n_past, 0,
                                                 context_params.n_batch, true, &n_past);
    if (eval != 0) {
        std::cerr << "multimodal prompt eval failed: " << eval << "\n";
        mtmd_input_chunks_free(chunks);
        for (auto * b : owned_bitmaps) mtmd_bitmap_free(b);
        mtmd_free(vision); llama_free(context); llama_model_free(model); llama_backend_free(); return 8;
    }

    const std::vector<std::string> options = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"};
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const float * logits = llama_get_logits_ith(context, -1);
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    if (!logits || vocab_size <= 0) { std::cerr << "answer-slot logits unavailable\n"; return 9; }
    std::vector<llama_token> token_ids;
    std::vector<float> option_logits;
    for (const auto & option : options) {
        llama_token token = LLAMA_TOKEN_NULL;
        const int32_t n = llama_tokenize(vocab, option.c_str(), option.size(), &token, 1, false, false);
        if (n != 1 || token < 0 || token >= vocab_size) {
            std::cerr << "answer choice is not one token: " << option << " (count=" << n << ")\n"; return 10;
        }
        if (std::find(token_ids.begin(), token_ids.end(), token) != token_ids.end()) {
            std::cerr << "duplicate answer token id\n"; return 11;
        }
        token_ids.push_back(token);
        option_logits.push_back(logits[token]);
    }
    const float max_logit = *std::max_element(option_logits.begin(), option_logits.end());
    std::vector<double> exp_scores;
    for (float value : option_logits) exp_scores.push_back(std::exp(static_cast<double>(value - max_logit)));
    const double denom = std::accumulate(exp_scores.begin(), exp_scores.end(), 0.0);
    std::cout << "{\"backend\":\"HIP\",\"device\":\"" << device_name
              << "\",\"device_description\":\"" << device_desc
              << "\",\"score_kind\":\"native_logits_at_answer_open_paren\",\"confidence\":\"uncalibrated\",\"images\":" << bitmaps.size()
              << ",\"options\":[";
    for (size_t i = 0; i < options.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"option\":\"" << options[i] << "\",\"token_id\":" << token_ids[i]
                  << ",\"logit\":" << option_logits[i] << ",\"probability\":" << exp_scores[i] / denom << '}';
    }
    std::cout << "],\"prompt_tokens_including_media\":" << n_past << "}\n";
    mtmd_input_chunks_free(chunks);
    for (auto * b : owned_bitmaps) mtmd_bitmap_free(b);
    mtmd_free(vision); llama_free(context); llama_model_free(model); llama_backend_free();
    return 0;
}
