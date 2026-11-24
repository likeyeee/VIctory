use super::*;

#[test]
fn table_cbv_observes_intervening_buffer_copy() {
    use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
    use windows::core::s;

    let Some(host) = crate::host::host() else {
        return;
    };
    let list = host.new_list().unwrap();
    let shader = br#"
        cbuffer Record : register(b0) { uint4 words[6]; };
        cbuffer OutputOffset : register(b1) { uint offset; };
        RWByteAddressBuffer result : register(u0);
        [RootSignature("DescriptorTable(CBV(b0)), UAV(u0), RootConstants(num32BitConstants=1, b1)")]
        [numthreads(1, 1, 1)]
        void main() { result.Store2(offset, words[4].xy); }
    "#;
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the input and blob stay alive through compilation and PSO creation.
    let (root, pso) = unsafe {
        D3DCompile(
            shader.as_ptr().cast(),
            shader.len(),
            s!("cbv_versions"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            None,
        )
        .unwrap();
        let code = code.unwrap();
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
        let pipeline = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        };
        let pso: ID3D12PipelineState = host.device.CreateComputePipelineState(&pipeline).unwrap();
        (root, pso)
    };

    #[repr(align(16))]
    struct Record([u32; 26]); // 96-byte CBV followed by eight unrelated bytes
    let mut record = Record([0; 26]);
    let mut buffer_source = [0u32; 2];
    let mut memory_source = [0u32; 2];
    let address = record.0.as_mut_ptr() as u64;
    let mut slots = [0u64; 4];
    let table = slots.as_mut_ptr() as u64;
    let heap = crate::descriptor::tests::placed_view_heap(table, 1);
    crate::descriptor::create_cbv(
        std::ptr::null(),
        &[address as u32, (address >> 32) as u32, 96, 0],
        table,
    );
    let native = &crate::descriptor::heap(heap).host().unwrap().heap;
    let (result, mapped) = host.shared_buffer(48).unwrap();
    let mut state = State::default();
    state.set_heaps(&[heap]);
    state.set_table(true, 0, table);
    let cbv = UsedResource {
        kind: ResourceType::Cbv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    // SAFETY: these objects remain live through both submissions.
    let (start, increment) = unsafe {
        list.list.SetDescriptorHeaps(&[Some(native.clone())]);
        list.list.SetComputeRootSignature(&root);
        list.list.SetPipelineState(&pso);
        list.list
            .SetComputeRootUnorderedAccessView(1, result.GetGPUVirtualAddress());
        (
            native.GetGPUDescriptorHandleForHeapStart().ptr,
            host.device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
        )
    };
    let mut sample = |offset| {
        let table_bytes = state.tables(&list, &[cbv], true)[0].1;
        let handle = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start + u64::from(table_bytes / 32) * u64::from(increment),
        };
        // SAFETY: the table owns this CBV and all six eight-byte outputs are disjoint.
        unsafe {
            list.list.SetComputeRootDescriptorTable(0, handle);
            list.list.SetComputeRoot32BitConstant(2, offset, 0);
            list.list.Dispatch(1, 1, 1);
        }
        table_bytes / 32
    };
    let initial = sample(0);
    // The first copy starts exactly at the CBV's end.
    list.defer_copy(
        record.0[24..].as_mut_ptr().cast(),
        buffer_source.as_ptr().cast(),
        8,
    );
    let unrelated = sample(8);
    // CopyBufferRegion and CopyMemoryToMemoryX both change words at 0x40 and 0x44.
    let dst = record.0[16..].as_mut_ptr();
    list.defer_copy(dst.cast(), buffer_source.as_ptr().cast(), 8);
    let after_buffer = sample(16);
    let reuse_buffer = sample(24);
    let dst = dst as usize;
    let src = memory_source.as_ptr() as usize;
    list.defer_title_copy(dst, 8, move || {
        // SAFETY: both live arrays contain the two words, including on resubmission.
        unsafe { std::ptr::copy(src as *const u32, dst as *mut u32, 2) };
    });
    let after_memory = sample(32);
    let reuse_memory = sample(40);
    assert_eq!(
        [unrelated, reuse_buffer, reuse_memory],
        [initial, after_buffer, after_memory]
    );
    eprintln!(
        "CBV slots: {initial} {unrelated} {after_buffer} {reuse_buffer} {after_memory} {reuse_memory}"
    );
    list.close();
    for (first, second, third) in [([3, 7], [11, 13], [17, 19]), ([23, 29], [31, 37], [41, 43])] {
        record.0[16..18].copy_from_slice(&first);
        buffer_source.copy_from_slice(&second);
        memory_source.copy_from_slice(&third);
        host.execute(&[&list]);
        // SAFETY: the fence completed and the mapped result contains six word pairs.
        let observed = unsafe { mapped.cast::<[[u32; 2]; 6]>().read() };
        eprintln!("GPU CBV reads: {observed:?}");
        assert_eq!(observed, [first, first, second, second, third, third]);
        assert_eq!(record.0[16..18], third);
        assert_eq!(record.0[24..26], second);
    }
    assert_eq!(state.dynamic.len(), 3);
    drop(state);
    crate::object::slot::<extern "system" fn(This) -> u32>(heap, 0x10)(heap);
}
