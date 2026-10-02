"""Minimal dispatch change to TeleOCRClient.stepping_two_step_extract at pinned 9921cffe.

The upstream body is retained in SOURCE_METHOD.txt/diff at packaging. Never monkeypatch
installed sources. Off calls superclass; pass-through exercises the same dispatcher.
"""
import copy
from .region_protocol import (eligible, image_binding, request, validate_response,
                              RunBlocked, ExpertTimeout, PauseAfterFallback)


def make_client_class(base):
    class HybridTeleOCRClient(base):
        def __init__(self, *args, hybrid_mode='off', formula_expert=None, emit=None, **kwargs):
            if hybrid_mode not in ('off', 'pass-through', 'on'): raise ValueError('Unknown adapter mode')
            super().__init__(*args, **kwargs)
            if self.backend != 'transformers' or self.batching_mode != 'stepping':
                raise RunBlocked('Only frozen Transformers stepping is supported')
            if self.incremental_priority or self.client.batch_size != 1:
                raise RunBlocked('Expected incremental_priority=False and batch_size=1')
            self.hybrid_mode, self.formula_expert = hybrid_mode, formula_expert
            self.emit = emit or (lambda event: None)
            self.context = None

        def set_context(self, run_id, page_id, input_sha256):
            self.context = {'run_id': run_id, 'page_id': page_id, 'input_sha256': input_sha256}

        def stepping_two_step_extract(self, images, priority=None, not_extract_list=None):
            if self.hybrid_mode == 'off':
                return super().stepping_two_step_extract(images, priority, not_extract_list)
            if self.context is None: raise RunBlocked('Input identity was not bound')
            if priority is not None: raise RunBlocked('Non-default priority unsupported in this revision')
            # Same upstream layout, preparation, flattening, slot assignment and postprocess.
            blocks_list = self.batch_layout_detect(images, priority)
            if len(blocks_list) != len(images): raise RunBlocked('Layout/page arity mismatch')
            all_images, all_prompts, all_params, all_indices = [], [], [], []
            prepared_inputs = self.helper.batch_prepare_for_extract(
                self.executor, images, blocks_list, not_extract_list)
            if len(prepared_inputs) != len(images): raise RunBlocked('Prepared/page arity mismatch')
            for img_idx, (block_images, prompts, params, indices) in enumerate(prepared_inputs):
                if not len(block_images) == len(prompts) == len(params) == len(indices):
                    raise RunBlocked('Prepared slot arity mismatch')
                if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(blocks_list[img_idx]) for i in indices):
                    raise RunBlocked('Invalid or duplicate original slot index')
                all_images.extend(block_images)
                all_prompts.extend(prompts)
                all_params.extend(params)
                all_indices.extend([(img_idx, idx) for idx in indices])
            metadata = [[{k: copy.deepcopy(v) for k, v in block.items() if k != 'content'} for block in blocks]
                        for blocks in blocks_list]
            outputs = self._hybrid_dispatch(images, blocks_list, all_images, all_prompts, all_params, all_indices, priority)
            if len(outputs) != len(all_indices): raise RunBlocked('Recognition result arity mismatch')
            for (img_idx, idx), output in zip(all_indices, outputs):
                if not isinstance(output, str): raise RunBlocked('Recognition output must be text')
                blocks_list[img_idx][idx].content = output
            after = [[{k: copy.deepcopy(v) for k, v in block.items() if k != 'content'} for block in blocks]
                     for blocks in blocks_list]
            if after != metadata: raise RunBlocked('Non-content block metadata changed')
            self.emit({'event': 'blocks_before_postprocess', **self.context, 'blocks': copy.deepcopy(blocks_list)})
            result = self.helper.batch_post_process(self.executor, blocks_list)
            self.emit({'event': 'blocks_after_postprocess', **self.context, 'blocks': copy.deepcopy(result)})
            return result

        def _hybrid_dispatch(self, pages, blocks, crops, prompts, params, indices, priority):
            if self.hybrid_mode == 'pass-through':
                outputs = self.client.batch_predict(crops, prompts, params, priority)
                for img_idx, index in indices:
                    self.emit({'event': 'route', **self.context, 'page_ordinal': img_idx,
                               'block_index': index, 'route': 'tele_native', 'reason': 'pass_through'})
                return outputs
            if self.formula_expert is None: raise RunBlocked('Expert was not initialized')
            outputs = [None] * len(indices)
            original = []
            for position, (img_idx, index) in enumerate(indices):
                allowed, reason = eligible(blocks[img_idx][index], crops[position])
                if not allowed:
                    original.append(position)
                    self.emit({'event': 'route', **self.context, 'page_ordinal': img_idx,
                               'block_index': index, 'route': 'tele_native', 'reason': reason})
                    continue
                sent = request(self.context['run_id'], self.context['page_id'], self.context['input_sha256'],
                               img_idx, index, crops[position])
                event = {'event': 'route', **{k: v for k, v in sent.items() if k != 'png_base64'},
                         'page_ordinal': img_idx, 'block_index': index,
                         'block': copy.deepcopy(blocks[img_idx][index]), 'rendered_page': image_binding(pages[img_idx]),
                         'prepared_crop': image_binding(crops[position])}
                timeout = False
                try:
                    response = self.formula_expert.recognize(sent)
                    event['expert_response'] = response
                    text, failure = validate_response(sent, response)
                except ExpertTimeout as exc:
                    # Transport raises only after the owned worker is confirmed stopped.
                    timeout, text, failure = True, None, 'timeout'
                    event['expert_error'] = str(exc)
                if failure is None:
                    outputs[position] = text
                    event['route'] = 'paddle_success'
                else:
                    fallback = self.client.batch_predict([crops[position]], [prompts[position]], [params[position]], priority)
                    if len(fallback) != 1 or not isinstance(fallback[0], str):
                        raise RunBlocked('Fallback result arity/type mismatch')
                    outputs[position] = fallback[0]
                    event.update(route='tele_fallback', reason=failure, tele_fallback_text=fallback[0])
                self.emit(event)
                if timeout:
                    # Preserve terminal Paddle failure and this slot's Tele result; do not send another request.
                    raise PauseAfterFallback('Expert timeout: current slot fallback saved; owner restart receipt required')
            if original:
                native = self.client.batch_predict([crops[i] for i in original], [prompts[i] for i in original],
                                                  [params[i] for i in original], priority)
                if len(native) != len(original): raise RunBlocked('Native result arity mismatch')
                for position, output in zip(original, native): outputs[position] = output
            return outputs

        async def aio_stepping_two_step_extract(self, *args, **kwargs):
            raise RunBlocked('Async path is not supported by this frozen adapter')

    return HybridTeleOCRClient
