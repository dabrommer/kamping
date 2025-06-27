// This file is part of KaMPIng.
//
// Copyright 2025 The KaMPIng Authors
//
// KaMPIng is free software : you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version. KaMPIng is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with KaMPIng.  If not, see <https://www.gnu.org/licenses/>.


#pragma once
#include <mdspan>


namespace kamping::adapter {

template <typename T, typename Extent, typename DataBufferType>
class MDSpanBuffer {

public:
    using value_type = T;

    explicit MDSpanBuffer(DataBufferType&& object) : _object(std::move(object)), _data() {unpack();}

    void unpack() {
        _mdspan = _object.underlying();
        _data = _mdspan.data_handle();
    }

    T* data() noexcept {
        return _data;
    }

    T const* data() const noexcept {
        return _data;
    }

    [[nodiscard]] size_t size() const {
        return _mdspan.size();
    }

private:
    DataBufferType _object;
    std::mdspan<T, Extent> _mdspan;
    T* _data;

};
    template<typename T, typename Extent, typename Accessor = std::default_accessor<T>>
    auto md_span_send(std::mdspan<T, Extent> data) {

        static_assert(std::is_same_v<Accessor, std::default_accessor<T>>,
                  "use std::default_accessor<T>");
            internal::GenericDataBuffer<
                decltype(data),
            internal::ParameterType,
            internal::ParameterType::send_recv_buf,
            internal::BufferModifiability::modifiable,
            internal::BufferOwnership::owning,
            internal::BufferType::in_out_buffer>
            buffer(data);
            return MDSpanBuffer<T, Extent, decltype(buffer)>(std::move(buffer));
    }
}



